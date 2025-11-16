// Function: sub_2A0B200
// Address: 0x2a0b200
//
void __fastcall sub_2A0B200(__int64 a1, __int64 a2, char a3, char a4)
{
  __int64 v6; // rax
  __int64 v7; // r15
  __int64 v8; // rax
  unsigned int v9; // eax
  int v10; // edx
  unsigned int v11; // ecx
  unsigned int v12; // eax
  unsigned int v13; // [rsp+8h] [rbp-68h]
  unsigned int v14; // [rsp+8h] [rbp-68h]
  _BOOL4 v15; // [rsp+Ch] [rbp-64h]
  unsigned int v16; // [rsp+10h] [rbp-60h] BYREF
  unsigned int v17; // [rsp+14h] [rbp-5Ch]
  unsigned int v18[2]; // [rsp+18h] [rbp-58h] BYREF
  _BYTE *v19; // [rsp+20h] [rbp-50h] BYREF
  __int64 v20; // [rsp+28h] [rbp-48h]
  _BYTE v21[64]; // [rsp+30h] [rbp-40h] BYREF

  v6 = sub_BC89C0(a1);
  if ( v6 )
  {
    v7 = v6;
    v8 = sub_BC89C0(a2);
    if ( v8 == v7 )
    {
      v19 = v21;
      v20 = 0x200000000LL;
      sub_BC8A70(v8, (__int64)&v19);
      if ( (_DWORD)v20 != 2 )
        goto LABEL_15;
      v9 = *(_DWORD *)v19;
      v10 = *((_DWORD *)v19 + 1);
      if ( a4 )
      {
        v9 = *((_DWORD *)v19 + 1);
        v10 = *(_DWORD *)v19;
      }
      if ( v9 )
      {
        if ( v10 )
        {
          if ( a3 )
          {
            if ( v9 <= v10 )
            {
              while ( v9 <= 0x7F && v10 >= 0 )
              {
                v10 *= 2;
                v9 *= 2;
              }
              v11 = v9 - 1;
              v10 = v10 + 1 - v9;
              v9 = 1;
            }
            else
            {
              v9 -= v10;
              v11 = v10;
              v10 = 0;
            }
          }
          else
          {
            v11 = v9;
            if ( v9 >= v10 )
              v10 = v9;
            v10 -= v9;
            v9 = 0;
          }
          v15 = v11;
        }
        else
        {
          v15 = 0;
          v11 = 1;
          v9 = 1;
        }
      }
      else
      {
        v15 = v10 != 0;
        v11 = 0;
      }
      if ( a4 )
      {
        v16 = v10;
        v17 = v11;
        v13 = v9;
        sub_BC8EC0(a2, &v16, 2, 0);
        v12 = v13;
        if ( !a3 )
          goto LABEL_15;
        v18[0] = v15;
      }
      else
      {
        v16 = v11;
        v17 = v10;
        v14 = v9;
        sub_BC8EC0(a2, &v16, 2, 0);
        if ( !a3 )
          goto LABEL_15;
        v18[0] = v14;
        v12 = v15;
      }
      v18[1] = v12;
      sub_BC8EC0(a1, v18, 2, 0);
LABEL_15:
      if ( v19 != v21 )
        _libc_free((unsigned __int64)v19);
    }
  }
}
