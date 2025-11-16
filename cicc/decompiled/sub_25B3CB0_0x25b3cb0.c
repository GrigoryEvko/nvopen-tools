// Function: sub_25B3CB0
// Address: 0x25b3cb0
//
void __fastcall sub_25B3CB0(_QWORD *a1)
{
  unsigned __int64 v2; // rdi
  __int64 v3; // rax
  __int64 *v4; // r12
  unsigned __int64 v5; // rax
  __int64 *v6; // rdx
  __int64 v7; // rax
  __int64 j; // r15
  unsigned __int8 *v9; // rbx
  int v10; // eax
  unsigned __int64 v11; // rax
  __int64 v12; // rcx
  unsigned __int64 v13; // rdi
  _QWORD *v14; // rax
  _QWORD *v15; // r8
  __int64 v16; // rsi
  __int64 v17; // rdx
  unsigned __int64 v18; // r12
  unsigned __int64 v19; // rdi
  __int64 v20; // r15
  bool v21; // r8
  __int64 v22; // rax
  _QWORD *v23; // rax
  __int64 *v24; // rdx
  __int64 k; // r12
  unsigned __int64 v26; // r12
  unsigned __int64 v27; // rdi
  __int64 *v28; // [rsp+0h] [rbp-C0h]
  char v29; // [rsp+8h] [rbp-B8h]
  unsigned __int64 v30; // [rsp+28h] [rbp-98h] BYREF
  __int64 v31; // [rsp+30h] [rbp-90h] BYREF
  __int64 v32; // [rsp+38h] [rbp-88h] BYREF
  unsigned __int64 v33; // [rsp+40h] [rbp-80h]
  __int64 *v34; // [rsp+48h] [rbp-78h]
  __int64 *v35; // [rsp+50h] [rbp-70h]
  __int64 v36; // [rsp+58h] [rbp-68h]
  char v37[8]; // [rsp+60h] [rbp-60h] BYREF
  int v38; // [rsp+68h] [rbp-58h] BYREF
  unsigned __int64 v39; // [rsp+70h] [rbp-50h]
  int *v40; // [rsp+78h] [rbp-48h]
  int *v41; // [rsp+80h] [rbp-40h]
  __int64 i; // [rsp+88h] [rbp-38h]

  v2 = a1[14];
  v34 = &v32;
  LODWORD(v32) = 0;
  v33 = 0;
  v35 = &v32;
  v36 = 0;
  if ( v2 )
  {
    v3 = sub_25AE1A0(v2, (__int64)&v32);
    v2 = v3;
    do
    {
      v4 = (__int64 *)v3;
      v3 = *(_QWORD *)(v3 + 16);
    }
    while ( v3 );
    v34 = v4;
    v5 = v2;
    do
    {
      v6 = (__int64 *)v5;
      v5 = *(_QWORD *)(v5 + 24);
    }
    while ( v5 );
    v7 = a1[17];
    v35 = v6;
    v33 = v2;
    v36 = v7;
    if ( v7 )
    {
      while ( 1 )
      {
        v38 = 0;
        v39 = 0;
        v40 = &v38;
        v41 = &v38;
        for ( i = 0; v4 != &v32; v4 = (__int64 *)sub_220EF30((__int64)v4) )
        {
          for ( j = *(_QWORD *)(v4[4] + 16); j; j = *(_QWORD *)(j + 8) )
          {
            v9 = *(unsigned __int8 **)(j + 24);
            v10 = *v9;
            if ( (unsigned __int8)v10 > 0x1Cu )
            {
              v11 = (unsigned int)(v10 - 34);
              if ( (unsigned __int8)v11 <= 0x33u )
              {
                v12 = 0x8000000000041LL;
                if ( _bittest64(&v12, v11) )
                {
                  if ( sub_B49200(*(_QWORD *)(j + 24)) )
                  {
                    v13 = *(_QWORD *)(*((_QWORD *)v9 + 5) + 72LL);
                    v14 = (_QWORD *)a1[14];
                    if ( !v14 )
                      goto LABEL_20;
                    v15 = a1 + 13;
                    do
                    {
                      while ( 1 )
                      {
                        v16 = v14[2];
                        v17 = v14[3];
                        if ( v14[4] >= v13 )
                          break;
                        v14 = (_QWORD *)v14[3];
                        if ( !v17 )
                          goto LABEL_18;
                      }
                      v15 = v14;
                      v14 = (_QWORD *)v14[2];
                    }
                    while ( v16 );
LABEL_18:
                    if ( a1 + 13 == v15 || v15[4] > v13 )
                    {
LABEL_20:
                      v30 = *(_QWORD *)(*((_QWORD *)v9 + 5) + 72LL);
                      sub_25B0A00((__int64)v37, &v30);
                    }
                  }
                }
              }
            }
          }
        }
        v18 = v33;
        while ( v18 )
        {
          sub_25AE610(*(_QWORD *)(v18 + 24));
          v19 = v18;
          v18 = *(_QWORD *)(v18 + 16);
          j_j___libc_free_0(v19);
        }
        v34 = &v32;
        v20 = (__int64)v40;
        v33 = 0;
        v35 = &v32;
        v36 = 0;
        if ( v40 != &v38 )
        {
          do
          {
            v23 = sub_25B3BB0(&v31, (__int64)&v32, (unsigned __int64 *)(v20 + 32));
            if ( v24 )
            {
              v21 = v23 || v24 == &v32 || *(_QWORD *)(v20 + 32) < (unsigned __int64)v24[4];
              v28 = v24;
              v29 = v21;
              v22 = sub_22077B0(0x28u);
              *(_QWORD *)(v22 + 32) = *(_QWORD *)(v20 + 32);
              sub_220F040(v29, v22, v28, &v32);
              ++v36;
            }
            v20 = sub_220EF30(v20);
          }
          while ( (int *)v20 != &v38 );
          for ( k = (__int64)v40; (int *)k != &v38; k = sub_220EF30(k) )
            sub_25B0AB0(a1, *(_QWORD *)(k + 32));
        }
        v26 = v39;
        while ( v26 )
        {
          sub_25AE610(*(_QWORD *)(v26 + 24));
          v27 = v26;
          v26 = *(_QWORD *)(v26 + 16);
          j_j___libc_free_0(v27);
        }
        if ( !v36 )
          break;
        v4 = v34;
      }
      v2 = v33;
    }
  }
  sub_25AE610(v2);
}
