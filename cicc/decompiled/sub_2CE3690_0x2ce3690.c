// Function: sub_2CE3690
// Address: 0x2ce3690
//
void __fastcall sub_2CE3690(_QWORD *a1, __int64 a2)
{
  __int64 v3; // rsi
  __int64 v4; // r12
  __int64 v6; // r13
  unsigned __int64 *v7; // rdx
  __int64 v8; // rdi
  unsigned __int64 *v9; // r9
  unsigned __int64 *v10; // r8
  unsigned __int64 *v11; // rax
  unsigned __int64 v12; // rsi
  unsigned __int64 v13; // rcx
  unsigned __int64 *v14; // rsi
  _BYTE *v15; // rsi
  unsigned __int64 v16; // rcx
  unsigned __int64 v17; // rax
  unsigned __int64 *v18; // [rsp+8h] [rbp-38h] BYREF
  unsigned __int64 *v19[5]; // [rsp+18h] [rbp-28h] BYREF

  v18 = (unsigned __int64 *)a2;
  v3 = *(_QWORD *)(a2 + 8);
  if ( *(_BYTE *)(v3 + 8) == 15 )
  {
    v4 = (__int64)(a1 + 53);
    v6 = sub_2CE23E0(*(_QWORD *)(a2 - 32), v3, a2, *(_BYTE *)(a2 + 2) & 1, (__int64)(a1 + 53));
    if ( v6 )
    {
      v7 = (unsigned __int64 *)a1[49];
      v8 = (__int64)v18;
      v9 = a1 + 48;
      if ( v7 )
      {
        v10 = a1 + 48;
        v11 = (unsigned __int64 *)a1[49];
        do
        {
          while ( 1 )
          {
            v12 = v11[2];
            v13 = v11[3];
            if ( v11[4] >= (unsigned __int64)v18 )
              break;
            v11 = (unsigned __int64 *)v11[3];
            if ( !v13 )
              goto LABEL_8;
          }
          v10 = v11;
          v11 = (unsigned __int64 *)v11[2];
        }
        while ( v12 );
LABEL_8:
        if ( v9 != v10 )
        {
          v14 = a1 + 48;
          if ( v10[4] <= (unsigned __int64)v18 )
          {
            do
            {
              while ( 1 )
              {
                v16 = v7[2];
                v17 = v7[3];
                if ( v7[4] >= (unsigned __int64)v18 )
                  break;
                v7 = (unsigned __int64 *)v7[3];
                if ( !v17 )
                  goto LABEL_18;
              }
              v14 = v7;
              v7 = (unsigned __int64 *)v7[2];
            }
            while ( v16 );
LABEL_18:
            if ( v9 == v14 || v14[4] > (unsigned __int64)v18 )
            {
              v19[0] = (unsigned __int64 *)&v18;
              v14 = sub_2CE35C0(a1 + 47, (__int64)v14, v19);
            }
            if ( (unsigned int)((__int64)(v14[6] - v14[5]) >> 3) )
              BUG();
            v8 = (__int64)v18;
          }
        }
      }
      sub_BD84D0(v8, v6);
      v15 = (_BYTE *)a1[54];
      v19[0] = v18;
      if ( v15 == (_BYTE *)a1[55] )
      {
        sub_249A840(v4, v15, v19);
      }
      else
      {
        if ( v15 )
        {
          *(_QWORD *)v15 = v18;
          v15 = (_BYTE *)a1[54];
        }
        a1[54] = v15 + 8;
      }
    }
  }
}
