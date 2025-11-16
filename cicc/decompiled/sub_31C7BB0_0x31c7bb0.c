// Function: sub_31C7BB0
// Address: 0x31c7bb0
//
__int64 __fastcall sub_31C7BB0(_QWORD *a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // rdx
  unsigned int v5; // r14d
  __int64 v6; // r13
  __int64 i; // r15
  char v8; // al
  _BYTE *v9; // rdi
  char v10; // al
  unsigned __int64 v11; // r9
  _QWORD *v12; // rax
  _QWORD *v13; // rsi
  __int64 v14; // rcx
  __int64 v15; // rdx
  _QWORD *v16; // rax
  _QWORD *v17; // rdx
  unsigned __int64 v18; // r9
  _QWORD *v19; // r14
  bool v20; // r10
  __int64 v21; // rdx
  __int64 v22; // rbx
  __int64 v23; // r13
  char v25; // al
  __int64 v26; // rax
  _BYTE *v27; // rsi
  __int64 v28; // [rsp+8h] [rbp-78h]
  __int64 v29; // [rsp+18h] [rbp-68h]
  _QWORD *v30; // [rsp+20h] [rbp-60h]
  __int64 v31; // [rsp+28h] [rbp-58h]
  _BYTE *v32; // [rsp+30h] [rbp-50h]
  char v33; // [rsp+30h] [rbp-50h]
  unsigned __int64 v34; // [rsp+38h] [rbp-48h]
  unsigned __int64 v35[7]; // [rsp+48h] [rbp-38h] BYREF

  v29 = (__int64)(a1 + 26);
  sub_31C5F30(a1[28]);
  a1[28] = 0;
  v30 = a1 + 27;
  a1[29] = a1 + 27;
  a1[30] = a1 + 27;
  v28 = (__int64)(a1 + 23);
  v3 = a1[23];
  a1[31] = 0;
  if ( a1[24] != v3 )
    a1[24] = v3;
  v4 = *(_QWORD *)(a2 + 40);
  a1[32] = a2;
  v5 = 0;
  v31 = a2 + 72;
  a1[22] = v4;
  v6 = *(_QWORD *)(a2 + 80);
  if ( v6 != a2 + 72 )
  {
    do
    {
      if ( !v6 )
        BUG();
      for ( i = *(_QWORD *)(v6 + 32); v6 + 24 != i; i = *(_QWORD *)(i + 8) )
      {
        if ( !i )
          BUG();
        v8 = *(_BYTE *)(i - 24);
        if ( v8 == 85 )
        {
          if ( **(_BYTE **)(i - 56) == 25 )
            v5 |= sub_31C6F90(a1, i - 24);
        }
        else if ( v8 == 61 )
        {
          v9 = *(_BYTE **)(i - 56);
          if ( *v9 == 3 )
          {
            v32 = *(_BYTE **)(i - 56);
            v10 = sub_CE8750(v9);
            v11 = i - 24;
            if ( v10 || (v25 = sub_CE87C0(v32), v11 = i - 24, v25) )
            {
              if ( !*(_QWORD *)(i - 8) )
              {
                v12 = (_QWORD *)a1[28];
                if ( !v12 )
                  goto LABEL_23;
                v13 = v30;
                do
                {
                  while ( 1 )
                  {
                    v14 = v12[2];
                    v15 = v12[3];
                    if ( v12[4] >= v11 )
                      break;
                    v12 = (_QWORD *)v12[3];
                    if ( !v15 )
                      goto LABEL_21;
                  }
                  v13 = v12;
                  v12 = (_QWORD *)v12[2];
                }
                while ( v14 );
LABEL_21:
                if ( v13 == v30 || v13[4] > v11 )
                {
LABEL_23:
                  v35[0] = v11;
                  v34 = v11;
                  v16 = sub_2D11AF0(v29, v35);
                  v18 = v34;
                  v19 = v17;
                  if ( v17 )
                  {
                    v20 = v16 || v30 == v17 || v34 < v17[4];
                    v33 = v20;
                    v26 = sub_22077B0(0x28u);
                    *(_QWORD *)(v26 + 32) = v35[0];
                    sub_220F040(v33, v26, v19, v30);
                    v18 = v34;
                    ++a1[31];
                  }
                  v35[0] = v18;
                  v27 = (_BYTE *)a1[24];
                  if ( v27 == (_BYTE *)a1[25] )
                  {
                    sub_249A840(v28, v27, v35);
                  }
                  else
                  {
                    if ( v27 )
                    {
                      *(_QWORD *)v27 = v18;
                      v27 = (_BYTE *)a1[24];
                    }
                    a1[24] = v27 + 8;
                  }
                }
                v5 = 1;
              }
            }
          }
        }
      }
      v6 = *(_QWORD *)(v6 + 8);
    }
    while ( v31 != v6 );
    v3 = a1[23];
  }
  v21 = (a1[24] - v3) >> 3;
  if ( (_DWORD)v21 )
  {
    v22 = 0;
    v23 = 8LL * (unsigned int)(v21 - 1);
    while ( 1 )
    {
      sub_B43D60(*(_QWORD **)(v3 + v22));
      if ( v22 == v23 )
        break;
      v3 = a1[23];
      v22 += 8;
    }
  }
  return v5;
}
