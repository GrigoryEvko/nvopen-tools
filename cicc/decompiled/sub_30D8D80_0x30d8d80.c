// Function: sub_30D8D80
// Address: 0x30d8d80
//
void __fastcall sub_30D8D80(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 *v3; // rsi
  __int64 v4; // rax
  unsigned __int64 v5; // rax
  unsigned int v7; // r13d
  __int64 v8; // r14
  __int64 v9; // rax
  __int64 v10; // rbx
  bool v11; // zf
  _QWORD *v12; // rax
  _QWORD *v13; // rdx
  _QWORD *v14; // rax
  __int64 i; // rdi
  __int64 v16; // rcx
  __int64 v17; // r8
  __int64 v18; // r9
  _QWORD *v19; // rdi
  unsigned int v20; // eax
  __int64 v21; // rdx
  __int64 v22; // rbx
  _QWORD *v23; // rax
  char v24; // dl
  unsigned __int64 v25; // rax
  __int64 v26; // r12
  int v27; // eax
  unsigned int v28; // r13d
  int v29; // ebx
  __int64 v30; // rax
  __int64 v31; // r14
  _QWORD *v32; // rax
  _QWORD *v33; // rdx
  __int64 v34; // rax
  __int64 j; // rdi
  __int64 v36; // rax
  unsigned __int64 v37; // rdx
  __int64 v38; // [rsp+18h] [rbp-98h]
  __int64 v39; // [rsp+28h] [rbp-88h]
  unsigned int v40; // [rsp+30h] [rbp-80h]
  int v41; // [rsp+34h] [rbp-7Ch]
  __int64 v43; // [rsp+40h] [rbp-70h] BYREF
  __int64 v44; // [rsp+48h] [rbp-68h] BYREF
  _QWORD *v45; // [rsp+50h] [rbp-60h] BYREF
  __int64 v46; // [rsp+58h] [rbp-58h]
  _QWORD v47[10]; // [rsp+60h] [rbp-50h] BYREF

  v3 = (__int64 *)(a2 + 48);
  v4 = *v3;
  v43 = a1;
  v5 = v4 & 0xFFFFFFFFFFFFFFF8LL;
  if ( (__int64 *)v5 != v3 )
  {
    if ( !v5 )
LABEL_60:
      BUG();
    v39 = v5 - 24;
    if ( (unsigned int)*(unsigned __int8 *)(v5 - 24) - 30 <= 0xA )
    {
      v41 = sub_B46E30(v5 - 24);
      if ( v41 )
      {
        v7 = 0;
        v8 = a1 + 264;
        while ( 1 )
        {
          v9 = sub_B46EC0(v39, v7);
          v10 = v9;
          if ( a3 == v9 )
            goto LABEL_12;
          v11 = *(_BYTE *)(a1 + 292) == 0;
          v45 = (_QWORD *)v9;
          if ( v11 )
          {
            if ( sub_C8CA60(v8, v9) )
              goto LABEL_12;
            v14 = v45;
            goto LABEL_15;
          }
          v12 = *(_QWORD **)(a1 + 272);
          v13 = &v12[*(unsigned int *)(a1 + 284)];
          if ( v12 == v13 )
          {
LABEL_14:
            v14 = (_QWORD *)v10;
LABEL_15:
            for ( i = v14[2]; i; i = *(_QWORD *)(i + 8) )
            {
              if ( (unsigned __int8)(**(_BYTE **)(i + 24) - 30) <= 0xAu )
                break;
            }
            if ( !sub_30D8A00(i, 0, &v43, &v45) )
              goto LABEL_12;
            v19 = v47;
            v47[0] = v10;
            v46 = 0x400000001LL;
            v20 = 1;
            v45 = v47;
            do
            {
              v21 = v20;
              v11 = *(_BYTE *)(a1 + 292) == 0;
              v22 = v19[v20 - 1];
              LODWORD(v46) = v20 - 1;
              if ( v11 )
                goto LABEL_31;
              v23 = *(_QWORD **)(a1 + 272);
              v16 = *(unsigned int *)(a1 + 284);
              v21 = (__int64)&v23[v16];
              if ( v23 != (_QWORD *)v21 )
              {
                while ( v22 != *v23 )
                {
                  if ( (_QWORD *)v21 == ++v23 )
                    goto LABEL_44;
                }
LABEL_24:
                v20 = v46;
                v19 = v45;
                continue;
              }
LABEL_44:
              if ( (unsigned int)v16 >= *(_DWORD *)(a1 + 280) )
              {
LABEL_31:
                sub_C8CC70(v8, v22, v21, v16, v17, v18);
                if ( !v24 )
                  goto LABEL_24;
              }
              else
              {
                v16 = (unsigned int)(v16 + 1);
                *(_DWORD *)(a1 + 284) = v16;
                *(_QWORD *)v21 = v22;
                ++*(_QWORD *)(a1 + 264);
              }
              v25 = *(_QWORD *)(v22 + 48) & 0xFFFFFFFFFFFFFFF8LL;
              if ( v25 == v22 + 48 )
                goto LABEL_24;
              if ( !v25 )
                goto LABEL_60;
              v26 = v25 - 24;
              if ( (unsigned int)*(unsigned __int8 *)(v25 - 24) - 30 > 0xA )
                goto LABEL_24;
              v27 = sub_B46E30(v26);
              if ( !v27 )
                goto LABEL_24;
              v38 = v8;
              v40 = v7;
              v28 = 0;
              v29 = v27;
              do
              {
                while ( 1 )
                {
                  v30 = sub_B46EC0(v26, v28);
                  v11 = *(_BYTE *)(a1 + 292) == 0;
                  v44 = v30;
                  v31 = v30;
                  if ( !v11 )
                    break;
                  if ( !sub_C8CA60(v38, v30) )
                  {
                    v34 = v44;
                    goto LABEL_47;
                  }
LABEL_42:
                  if ( v29 == ++v28 )
                    goto LABEL_43;
                }
                v32 = *(_QWORD **)(a1 + 272);
                v33 = &v32[*(unsigned int *)(a1 + 284)];
                if ( v32 != v33 )
                {
                  while ( v31 != *v32 )
                  {
                    if ( v33 == ++v32 )
                      goto LABEL_46;
                  }
                  goto LABEL_42;
                }
LABEL_46:
                v34 = v31;
LABEL_47:
                for ( j = *(_QWORD *)(v34 + 16); j; j = *(_QWORD *)(j + 8) )
                {
                  if ( (unsigned __int8)(**(_BYTE **)(j + 24) - 30) <= 0xAu )
                    break;
                }
                if ( !sub_30D8A00(j, 0, &v43, &v44) )
                  goto LABEL_42;
                v36 = (unsigned int)v46;
                v16 = HIDWORD(v46);
                v37 = (unsigned int)v46 + 1LL;
                if ( v37 > HIDWORD(v46) )
                {
                  sub_C8D5F0((__int64)&v45, v47, v37, 8u, v17, v18);
                  v36 = (unsigned int)v46;
                }
                ++v28;
                v45[v36] = v31;
                LODWORD(v46) = v46 + 1;
              }
              while ( v29 != v28 );
LABEL_43:
              v7 = v40;
              v8 = v38;
              v20 = v46;
              v19 = v45;
            }
            while ( v20 );
            if ( v19 == v47 )
              goto LABEL_12;
            _libc_free((unsigned __int64)v19);
            if ( v41 == ++v7 )
              return;
          }
          else
          {
            while ( v10 != *v12 )
            {
              if ( v13 == ++v12 )
                goto LABEL_14;
            }
LABEL_12:
            if ( v41 == ++v7 )
              return;
          }
        }
      }
    }
  }
}
