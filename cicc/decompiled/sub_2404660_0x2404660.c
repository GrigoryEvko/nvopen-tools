// Function: sub_2404660
// Address: 0x2404660
//
void __fastcall sub_2404660(__int64 a1, __int64 a2)
{
  __int64 *v2; // rbx
  unsigned int v5; // eax
  __int64 v6; // r13
  __int64 v7; // r12
  double v8; // xmm0_8
  unsigned int v9; // r13d
  unsigned int v10; // esi
  _QWORD *v11; // r9
  __int64 *v12; // r8
  __int64 v13; // rdx
  __int64 v14; // r11
  _QWORD *v15; // rax
  __int64 v16; // rdi
  unsigned int *v17; // rax
  __int64 v18; // rax
  __int64 v19; // r12
  unsigned int v20; // esi
  __int64 v21; // rdx
  __int64 v22; // r11
  unsigned int v23; // r13d
  _QWORD *v24; // rax
  __int64 v25; // rdi
  unsigned int *v26; // rax
  __int64 *v27; // rcx
  int v28; // eax
  int v29; // edi
  __int64 *v30; // rcx
  int v31; // edi
  int v32; // eax
  __int64 v33; // [rsp+0h] [rbp-C0h]
  __int64 v34; // [rsp+8h] [rbp-B8h]
  int v35; // [rsp+10h] [rbp-B0h]
  int v36; // [rsp+10h] [rbp-B0h]
  unsigned int v37; // [rsp+20h] [rbp-A0h]
  unsigned int v38; // [rsp+20h] [rbp-A0h]
  __int64 *v39; // [rsp+38h] [rbp-88h]
  int v40; // [rsp+48h] [rbp-78h] BYREF
  int v41; // [rsp+4Ch] [rbp-74h] BYREF
  __int64 v42; // [rsp+50h] [rbp-70h] BYREF
  __int64 v43; // [rsp+58h] [rbp-68h] BYREF
  _QWORD v44[12]; // [rsp+60h] [rbp-60h] BYREF

  v2 = **(__int64 ***)a1;
  v39 = &v2[*(unsigned int *)(*(_QWORD *)a1 + 8LL)];
  if ( v39 != v2 )
  {
    while ( 1 )
    {
      v6 = *v2;
      v7 = *(_QWORD *)(a1 + 8);
      v40 = -1;
      v41 = -1;
      v42 = v6;
      if ( !(unsigned __int8)sub_23FAB40(v6, &v40, &v41) )
        goto LABEL_5;
      v43 = v6;
      v8 = 1000000.0 * *(double *)&qword_4FE2C28;
      v9 = v40;
      v37 = v41;
      if ( 1000000.0 * *(double *)&qword_4FE2C28 < 9.223372036854776e18 )
        break;
      v5 = sub_F02DD0((unsigned int)(int)(v8 - 9.223372036854776e18) ^ 0x8000000000000000LL, 0xF4240u);
      if ( v9 < v5 )
      {
LABEL_4:
        if ( v37 >= v5 )
        {
          sub_24044F0((__int64)v44, v7 + 168, &v43);
          v20 = *(_DWORD *)(v7 + 256);
          v11 = v44;
          v33 = v7 + 232;
          v12 = &v43;
          if ( v20 )
          {
            v21 = v43;
            v22 = *(_QWORD *)(v7 + 240);
            v23 = (v20 - 1) & (((unsigned int)v43 >> 9) ^ ((unsigned int)v43 >> 4));
            v24 = (_QWORD *)(v22 + 16LL * v23);
            v25 = *v24;
            if ( *v24 == v43 )
            {
LABEL_19:
              v26 = (unsigned int *)(v24 + 1);
LABEL_20:
              *v26 = v37;
              goto LABEL_13;
            }
            v36 = 1;
            v30 = 0;
            while ( v25 != -4096 )
            {
              if ( v25 == -8192 && !v30 )
                v30 = v24;
              v23 = (v20 - 1) & (v36 + v23);
              v24 = (_QWORD *)(v22 + 16LL * v23);
              v25 = *v24;
              if ( v43 == *v24 )
                goto LABEL_19;
              ++v36;
            }
            if ( !v30 )
              v30 = v24;
            v44[0] = v30;
            v32 = *(_DWORD *)(v7 + 248);
            ++*(_QWORD *)(v7 + 232);
            v31 = v32 + 1;
            if ( 4 * (v32 + 1) < 3 * v20 )
            {
              if ( v20 - *(_DWORD *)(v7 + 252) - v31 <= v20 >> 3 )
              {
LABEL_32:
                sub_23FF570(v33, v20);
                sub_23FDC90(v33, &v43, v44);
                v21 = v43;
                v30 = (__int64 *)v44[0];
                v31 = *(_DWORD *)(v7 + 248) + 1;
              }
              *(_DWORD *)(v7 + 248) = v31;
              if ( *v30 != -4096 )
                --*(_DWORD *)(v7 + 252);
              *v30 = v21;
              v26 = (unsigned int *)(v30 + 1);
              *((_DWORD *)v30 + 2) = -1;
              goto LABEL_20;
            }
          }
          else
          {
            v44[0] = 0;
            ++*(_QWORD *)(v7 + 232);
          }
          v20 *= 2;
          goto LABEL_32;
        }
LABEL_5:
        ++v2;
        sub_23FE340(*(__int64 **)(*(_QWORD *)(a1 + 8) + 40LL), &v42);
        if ( v39 == v2 )
          return;
      }
      else
      {
LABEL_9:
        sub_24044F0((__int64)v44, v7 + 136, &v43);
        v10 = *(_DWORD *)(v7 + 256);
        v11 = v44;
        v34 = v7 + 232;
        v12 = &v43;
        if ( !v10 )
        {
          v44[0] = 0;
          ++*(_QWORD *)(v7 + 232);
          goto LABEL_49;
        }
        v13 = v43;
        v14 = *(_QWORD *)(v7 + 240);
        v38 = (v10 - 1) & (((unsigned int)v43 >> 9) ^ ((unsigned int)v43 >> 4));
        v15 = (_QWORD *)(v14 + 16LL * v38);
        v16 = *v15;
        if ( v43 != *v15 )
        {
          v35 = 1;
          v27 = 0;
          while ( v16 != -4096 )
          {
            if ( v16 == -8192 && !v27 )
              v27 = v15;
            v38 = (v10 - 1) & (v38 + v35);
            v15 = (_QWORD *)(v14 + 16LL * v38);
            v16 = *v15;
            if ( v43 == *v15 )
              goto LABEL_11;
            ++v35;
          }
          if ( !v27 )
            v27 = v15;
          v44[0] = v27;
          v28 = *(_DWORD *)(v7 + 248);
          ++*(_QWORD *)(v7 + 232);
          v29 = v28 + 1;
          if ( 4 * (v28 + 1) < 3 * v10 )
          {
            if ( v10 - *(_DWORD *)(v7 + 252) - v29 > v10 >> 3 )
            {
LABEL_27:
              *(_DWORD *)(v7 + 248) = v29;
              if ( *v27 != -4096 )
                --*(_DWORD *)(v7 + 252);
              *v27 = v13;
              v17 = (unsigned int *)(v27 + 1);
              *((_DWORD *)v27 + 2) = -1;
              goto LABEL_12;
            }
LABEL_50:
            sub_23FF570(v34, v10);
            sub_23FDC90(v34, &v43, v44);
            v13 = v43;
            v27 = (__int64 *)v44[0];
            v29 = *(_DWORD *)(v7 + 248) + 1;
            goto LABEL_27;
          }
LABEL_49:
          v10 *= 2;
          goto LABEL_50;
        }
LABEL_11:
        v17 = (unsigned int *)(v15 + 1);
LABEL_12:
        *v17 = v9;
LABEL_13:
        v18 = *(unsigned int *)(a2 + 24);
        v19 = v42;
        if ( v18 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 28) )
        {
          sub_C8D5F0(a2 + 16, (const void *)(a2 + 32), v18 + 1, 8u, (__int64)v12, (__int64)v11);
          v18 = *(unsigned int *)(a2 + 24);
        }
        ++v2;
        *(_QWORD *)(*(_QWORD *)(a2 + 16) + 8 * v18) = v19;
        ++*(_DWORD *)(a2 + 24);
        if ( v39 == v2 )
          return;
      }
    }
    v5 = sub_F02DD0((unsigned int)(int)v8, 0xF4240u);
    if ( v9 >= v5 )
      goto LABEL_9;
    goto LABEL_4;
  }
}
