// Function: sub_297C090
// Address: 0x297c090
//
_BOOL8 __fastcall sub_297C090(
        _QWORD *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        int a7,
        __int64 a8,
        __int64 a9)
{
  __int64 v9; // r15
  __int64 v10; // rsi
  __int64 v11; // rdi
  __int64 v12; // rax
  unsigned int v13; // edx
  __int64 *v14; // rbx
  __int64 v15; // rcx
  __int64 v16; // rsi
  __int64 v17; // r13
  __int64 v18; // rbx
  _BYTE *v19; // r15
  _BYTE *v20; // rax
  __int64 v21; // r12
  __int64 v22; // rdx
  __int64 v23; // rsi
  _QWORD *v24; // r15
  __int16 v25; // ax
  int i; // r9d
  __int64 v28; // rax
  __int64 v29; // rsi
  unsigned int v30; // edx
  __int64 v31; // rcx
  _QWORD *v32; // r8
  __int64 v33; // rsi
  __int64 v34; // rax
  __int64 v35; // r15
  __int64 v36; // rbx
  __int64 v37; // r13
  int v38; // ecx
  int v39; // r9d
  __int64 v40; // [rsp+10h] [rbp-70h]
  __int64 v41; // [rsp+18h] [rbp-68h]
  __int64 v43; // [rsp+28h] [rbp-58h]
  __int64 v44; // [rsp+28h] [rbp-58h]
  __int64 v45; // [rsp+30h] [rbp-50h]
  __int64 v47; // [rsp+40h] [rbp-40h]
  bool v48; // [rsp+4Bh] [rbp-35h]

  v9 = *(_QWORD *)(*a1 + 160LL);
  v47 = *(_QWORD *)(*a1 + 152LL);
  if ( v9 != v47 )
  {
    while ( 1 )
    {
      v10 = *(_QWORD *)(v9 - 8);
      v11 = *(_QWORD *)(a2 + 8);
      v12 = *(unsigned int *)(a2 + 24);
      if ( !(_DWORD)v12 )
        goto LABEL_3;
      v13 = (v12 - 1) & (((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4));
      v14 = (__int64 *)(v11 + 88LL * v13);
      v15 = *v14;
      if ( v10 != *v14 )
      {
        for ( i = 1; ; ++i )
        {
          if ( v15 == -4096 )
            goto LABEL_3;
          v13 = (v12 - 1) & (i + v13);
          v14 = (__int64 *)(v11 + 88LL * v13);
          v15 = *v14;
          if ( v10 == *v14 )
            break;
        }
      }
      if ( v14 != (__int64 *)(v11 + 88 * v12) )
      {
        if ( *((_DWORD *)v14 + 4) )
        {
          if ( (unsigned __int8)sub_B19720(*(_QWORD *)(*a1 + 8LL), v10, *(_QWORD *)(*(_QWORD *)(a1[1] + 32LL) + 40LL)) )
          {
            v16 = v14[1];
            v17 = v16 + 8LL * *((unsigned int *)v14 + 4);
            v45 = v16;
            if ( v16 != v17 )
              break;
          }
        }
      }
LABEL_3:
      v9 -= 8;
      if ( v47 == v9 )
        return 0;
    }
    v18 = a9;
    v41 = v9;
    while ( 1 )
    {
      v21 = *(_QWORD *)(v17 - 8);
      v48 = sub_297BA30(v18, v21, a7);
      if ( v48 )
      {
        if ( a7 == 2 )
        {
          v22 = *(_QWORD *)(v21 + 8);
          v23 = *(_QWORD *)(v18 + 8);
        }
        else
        {
          v22 = *(_QWORD *)(v21 + 56);
          v23 = *(_QWORD *)(v18 + 56);
        }
        v24 = sub_DCC810(*(__int64 **)(a8 + 16), v23, v22, 0, 0);
        v43 = *(_QWORD *)(v18 + 32);
        if ( !sub_D96A50((__int64)v24) )
        {
          v25 = *((_WORD *)v24 + 12);
          if ( v25 == 15 )
          {
            v19 = (_BYTE *)*(v24 - 1);
            goto LABEL_12;
          }
          if ( !v25 )
          {
            v19 = (_BYTE *)v24[4];
LABEL_12:
            if ( v19 )
            {
              v20 = *(_BYTE **)(v18 + 64);
              if ( !v20 || *v19 == 17 )
              {
                *(_QWORD *)(v18 + 64) = v19;
                *(_QWORD *)(v18 + 40) = v21;
                *(_DWORD *)(v18 + 48) = a7;
                if ( *v19 == 17 )
                  return v48;
              }
              else if ( *v20 == 17 )
              {
                return v48;
              }
            }
            goto LABEL_16;
          }
          v28 = *(unsigned int *)(a8 + 200);
          v29 = *(_QWORD *)(a8 + 184);
          if ( (_DWORD)v28 )
          {
            v30 = (v28 - 1) & (((unsigned int)v24 >> 9) ^ ((unsigned int)v24 >> 4));
            v31 = v29 + 72LL * v30;
            v32 = *(_QWORD **)v31;
            if ( *(_QWORD **)v31 == v24 )
            {
LABEL_33:
              if ( v31 != v29 + 72 * v28 )
              {
                v33 = *(_QWORD *)(v31 + 40);
                v34 = *(unsigned int *)(v31 + 48);
                if ( v33 != v33 + 8 * v34 )
                {
                  v35 = v43;
                  v44 = v18;
                  v36 = v33 + 8 * v34;
                  v40 = v17;
                  do
                  {
                    v37 = *(_QWORD *)(v36 - 8);
                    if ( (unsigned __int8)sub_B19DB0(*(_QWORD *)(a8 + 8), v37, v35) )
                    {
                      v19 = (_BYTE *)v37;
                      v17 = v40;
                      v18 = v44;
                      goto LABEL_12;
                    }
                    v36 -= 8;
                  }
                  while ( v33 != v36 );
                  v17 = v40;
                  v18 = v44;
                }
              }
            }
            else
            {
              v38 = 1;
              while ( v32 != (_QWORD *)-4096LL )
              {
                v39 = v38 + 1;
                v30 = (v28 - 1) & (v38 + v30);
                v31 = v29 + 72LL * v30;
                v32 = *(_QWORD **)v31;
                if ( v24 == *(_QWORD **)v31 )
                  goto LABEL_33;
                v38 = v39;
              }
            }
          }
        }
      }
LABEL_16:
      v17 -= 8;
      if ( v45 == v17 )
      {
        v9 = v41;
        goto LABEL_3;
      }
    }
  }
  return 0;
}
