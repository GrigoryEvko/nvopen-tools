// Function: sub_2B7C8C0
// Address: 0x2b7c8c0
//
__int64 ***__fastcall sub_2B7C8C0(__int64 **a1, __int64 a2, int *a3, unsigned __int64 a4)
{
  __int64 ***v4; // r13
  __int64 v7; // r8
  __int64 v8; // r9
  __int64 v10; // rdx
  __int64 v11; // rax
  unsigned int v12; // edx
  unsigned int v13; // edx
  unsigned int v14; // eax
  __int64 v15; // rcx
  _DWORD *v16; // rcx
  int *v17; // rdx
  void *s; // [rsp+10h] [rbp-70h] BYREF
  __int64 v19; // [rsp+18h] [rbp-68h]
  _DWORD v20[24]; // [rsp+20h] [rbp-60h] BYREF

  v4 = (__int64 ***)a2;
  if ( (_DWORD)a4 != *(_DWORD *)(*(_QWORD *)(a2 + 8) + 32LL) )
  {
    if ( &a3[a4] == sub_2B09020(a3, (__int64)&a3[a4], a4) )
    {
      if ( !(_BYTE)v7 )
      {
        s = v20;
        v19 = 0xC00000000LL;
        if ( (unsigned int)a4 > 0xC )
        {
          sub_C8D5F0((__int64)&s, v20, (unsigned int)a4, 4u, v7, v8);
          memset(s, 255, 4LL * (unsigned int)a4);
          LODWORD(v19) = a4;
          v16 = s;
        }
        else
        {
          if ( (_DWORD)a4 )
          {
            v10 = 4LL * (unsigned int)a4;
            if ( v10 )
            {
              if ( (unsigned int)v10 < 8 )
              {
                if ( ((4 * (_BYTE)a4) & 4) != 0 )
                {
                  v20[0] = -1;
                  *(_DWORD *)((char *)&v20[-1] + (unsigned int)v10) = -1;
                }
                else if ( (_DWORD)v10 )
                {
                  LOBYTE(v20[0]) = -1;
                }
              }
              else
              {
                v11 = (unsigned int)v10;
                v12 = v10 - 1;
                *(_QWORD *)((char *)&v20[-2] + v11) = -1;
                if ( v12 >= 8 )
                {
                  v13 = v12 & 0xFFFFFFF8;
                  v14 = 0;
                  do
                  {
                    v15 = v14;
                    v14 += 8;
                    *(_QWORD *)((char *)v20 + v15) = -1;
                  }
                  while ( v14 < v13 );
                }
              }
            }
          }
          LODWORD(v19) = a4;
          v16 = v20;
        }
        if ( (_DWORD)a4 )
        {
          v17 = a3;
          do
          {
            if ( *v17 != -1 )
            {
              v16[*v17] = *v17;
              v16 = s;
            }
            ++v17;
          }
          while ( v17 != &a3[(unsigned int)(a4 - 1) + 1] );
        }
        v4 = sub_2B7C3E0(*a1, a2, 0, (__int64)v16, (unsigned int)v19);
        if ( s != v20 )
          _libc_free((unsigned __int64)s);
      }
    }
    else
    {
      return sub_2B7C3E0(*a1, a2, 0, (__int64)a3, a4);
    }
  }
  return v4;
}
