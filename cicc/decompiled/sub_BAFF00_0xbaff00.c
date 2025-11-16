// Function: sub_BAFF00
// Address: 0xbaff00
//
__int64 __fastcall sub_BAFF00(__int64 a1, _QWORD *a2)
{
  __int64 result; // rax
  _QWORD *v3; // r15
  const void *v4; // r12
  size_t v5; // r13
  __int64 v6; // r14
  size_t v7; // rbx
  size_t v8; // rdx
  int v9; // eax
  size_t v10; // rcx
  size_t v11; // rdx
  int v12; // eax
  __int64 v13; // r8
  _QWORD *v14; // rax
  __int64 v15; // r12
  __int64 v16; // rcx
  __int64 v17; // rdx
  _QWORD *v18; // rax
  __int64 v19; // rdx
  _BOOL8 v20; // rdi
  __int64 v21; // rax
  __int64 v22; // rax
  _QWORD *v23; // rdx
  _QWORD *v24; // r11
  __int64 v25; // r8
  _BOOL8 v26; // rdi
  size_t v27; // rcx
  size_t v28; // rdx
  unsigned int v29; // eax
  size_t v30; // [rsp+8h] [rbp-88h]
  __int64 v31; // [rsp+10h] [rbp-80h]
  _QWORD *v32; // [rsp+18h] [rbp-78h]
  __int64 v33; // [rsp+20h] [rbp-70h]
  __int64 v34; // [rsp+20h] [rbp-70h]
  __int64 *v35; // [rsp+28h] [rbp-68h]
  size_t v36; // [rsp+30h] [rbp-60h]
  _QWORD *v37; // [rsp+30h] [rbp-60h]
  _QWORD *v38; // [rsp+30h] [rbp-60h]
  __int64 v39; // [rsp+30h] [rbp-60h]
  __int64 v40; // [rsp+30h] [rbp-60h]
  _QWORD *v41; // [rsp+30h] [rbp-60h]
  _QWORD *v42; // [rsp+30h] [rbp-60h]
  __int64 v43; // [rsp+40h] [rbp-50h]
  unsigned __int64 v44; // [rsp+48h] [rbp-48h]
  _QWORD *v45; // [rsp+50h] [rbp-40h]
  __int64 *v46; // [rsp+58h] [rbp-38h]

  result = a1 + 8;
  v31 = a1 + 8;
  v32 = *(_QWORD **)(a1 + 24);
  v45 = a2 + 1;
  if ( v32 != (_QWORD *)(a1 + 8) )
  {
    do
    {
      v44 = v32[4];
      v35 = (__int64 *)v32[8];
      v46 = (__int64 *)v32[7];
      if ( v35 == v46 )
        goto LABEL_35;
      do
      {
        v3 = a2 + 1;
        v4 = *(const void **)(*v46 + 24);
        v5 = *(_QWORD *)(*v46 + 32);
        v43 = *v46;
        v6 = a2[2];
        if ( !v6 )
          goto LABEL_40;
        do
        {
          while ( 1 )
          {
            v7 = *(_QWORD *)(v6 + 40);
            v8 = v7;
            if ( v5 <= v7 )
              v8 = v5;
            if ( v8 )
            {
              v9 = memcmp(*(const void **)(v6 + 32), v4, v8);
              if ( v9 )
                break;
            }
            if ( v5 != v7 && v5 > v7 )
            {
              v6 = *(_QWORD *)(v6 + 24);
              goto LABEL_13;
            }
LABEL_5:
            v3 = (_QWORD *)v6;
            v6 = *(_QWORD *)(v6 + 16);
            if ( !v6 )
              goto LABEL_14;
          }
          if ( v9 >= 0 )
            goto LABEL_5;
          v6 = *(_QWORD *)(v6 + 24);
LABEL_13:
          ;
        }
        while ( v6 );
LABEL_14:
        if ( v3 == v45 )
          goto LABEL_40;
        v10 = v3[5];
        v11 = v10;
        if ( v5 <= v10 )
          v11 = v5;
        if ( v11 )
        {
          v36 = v3[5];
          v12 = memcmp(v4, (const void *)v3[4], v11);
          v10 = v36;
          if ( v12 )
          {
            if ( v12 >= 0 )
              goto LABEL_21;
LABEL_40:
            v38 = v3;
            v21 = sub_22077B0(96);
            *(_QWORD *)(v21 + 32) = v4;
            v3 = (_QWORD *)v21;
            *(_QWORD *)(v21 + 40) = v5;
            *(_DWORD *)(v21 + 56) = 0;
            *(_QWORD *)(v21 + 64) = 0;
            *(_QWORD *)(v21 + 72) = v21 + 56;
            *(_QWORD *)(v21 + 80) = v21 + 56;
            *(_QWORD *)(v21 + 88) = 0;
            v34 = v21 + 56;
            v22 = sub_BAFA90(a2, v38, v21 + 32);
            v24 = v23;
            if ( v23 )
            {
              v25 = v34;
              if ( v45 == v23 || v22 )
              {
                v26 = 1;
              }
              else
              {
                v28 = v23[5];
                v27 = v28;
                if ( v5 <= v28 )
                  v28 = v5;
                if ( v28
                  && (v30 = v27,
                      v42 = v24,
                      v29 = memcmp(v4, (const void *)v24[4], v28),
                      v24 = v42,
                      v25 = v34,
                      v27 = v30,
                      v29) )
                {
                  v26 = v29 >> 31;
                }
                else
                {
                  v26 = v5 < v27;
                  if ( v5 == v27 )
                    v26 = 0;
                }
              }
              v39 = v25;
              sub_220F040(v26, v3, v24, v45);
              v13 = v39;
              ++a2[5];
            }
            else
            {
              v40 = v22;
              j_j___libc_free_0(v3, 96);
              v13 = v40 + 56;
              v3 = (_QWORD *)v40;
            }
            goto LABEL_22;
          }
        }
        if ( v5 != v10 && v5 < v10 )
          goto LABEL_40;
LABEL_21:
        v13 = (__int64)(v3 + 7);
LABEL_22:
        v14 = (_QWORD *)v3[8];
        v15 = v13;
        if ( !v14 )
          goto LABEL_29;
        do
        {
          while ( 1 )
          {
            v16 = v14[2];
            v17 = v14[3];
            if ( v44 <= v14[4] )
              break;
            v14 = (_QWORD *)v14[3];
            if ( !v17 )
              goto LABEL_27;
          }
          v15 = (__int64)v14;
          v14 = (_QWORD *)v14[2];
        }
        while ( v16 );
LABEL_27:
        if ( v15 == v13 || v44 < *(_QWORD *)(v15 + 32) )
        {
LABEL_29:
          v37 = (_QWORD *)v15;
          v33 = v13;
          v15 = sub_22077B0(48);
          *(_QWORD *)(v15 + 40) = 0;
          *(_QWORD *)(v15 + 32) = v44;
          v18 = sub_BAFE00(v3 + 6, v37, (unsigned __int64 *)(v15 + 32));
          if ( v19 )
          {
            v20 = v33 == v19 || v18 || v44 < *(_QWORD *)(v19 + 32);
            sub_220F040(v20, v15, v19, v33);
            ++v3[11];
          }
          else
          {
            v41 = v18;
            j_j___libc_free_0(v15, 48);
            v15 = (__int64)v41;
          }
        }
        ++v46;
        *(_QWORD *)(v15 + 40) = v43;
      }
      while ( v35 != v46 );
LABEL_35:
      result = sub_220EF30(v32);
      v32 = (_QWORD *)result;
    }
    while ( v31 != result );
  }
  return result;
}
