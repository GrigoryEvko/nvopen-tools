// Function: sub_25391D0
// Address: 0x25391d0
//
__int64 __fastcall sub_25391D0(
        __int64 a1,
        unsigned __int8 (__fastcall *a2)(__int64, _QWORD, _QWORD, _QWORD, _QWORD),
        __int64 a3,
        int a4)
{
  unsigned int v4; // r8d
  int v5; // eax
  unsigned int v7; // ebx
  __int64 v8; // r14
  __int64 v10; // rax
  __int64 v11; // r15
  char v12; // r14
  __int64 v13; // rax
  __int64 v15; // [rsp+10h] [rbp-40h]
  __int64 *v16; // [rsp+18h] [rbp-38h]

  v4 = 0;
  v5 = *(_DWORD *)(a1 + 100);
  if ( v5 )
  {
    if ( v5 != 255 )
    {
      v7 = 1;
      v16 = (__int64 *)(a1 + 104);
      do
      {
        if ( (v7 & a4) == 0 )
        {
          v8 = *v16;
          if ( *v16 )
          {
            if ( *(_QWORD *)(v8 + 120) )
            {
              v13 = v8 + 88;
              v11 = *(_QWORD *)(v8 + 104);
              v12 = 0;
            }
            else
            {
              v10 = *(unsigned int *)(v8 + 8);
              v11 = *(_QWORD *)v8;
              v12 = 1;
              v13 = v11 + 24 * v10;
            }
            v15 = v13;
            while ( v12 )
            {
              if ( v15 == v11 )
                goto LABEL_6;
              if ( !a2(a3, *(_QWORD *)v11, *(_QWORD *)(v11 + 8), *(unsigned int *)(v11 + 16), v7) )
                return 0;
              v11 += 24;
            }
            while ( v15 != v11 )
            {
              if ( !a2(a3, *(_QWORD *)(v11 + 32), *(_QWORD *)(v11 + 40), *(unsigned int *)(v11 + 48), v7) )
                return 0;
              v11 = sub_220EF30(v11);
            }
          }
        }
LABEL_6:
        ++v16;
        v7 *= 2;
      }
      while ( v16 != (__int64 *)(a1 + 168) );
    }
    return 1;
  }
  return v4;
}
