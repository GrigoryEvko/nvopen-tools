// Function: sub_2F640D0
// Address: 0x2f640d0
//
void __fastcall sub_2F640D0(_QWORD *a1, __int64 a2, _QWORD *a3)
{
  __int64 v3; // rax
  __int64 v5; // r15
  __int64 v6; // r12
  __int64 v7; // rax
  __int64 v8; // rbx
  unsigned __int64 v9; // r13
  __int64 v10; // rdx
  __int64 v11; // rdx
  __int64 *v12; // rax
  __int64 v13; // rsi
  unsigned int v14; // edi
  unsigned int v15; // ecx
  __int64 v16; // r11
  __int64 v17; // rax
  __int64 v18; // rdi
  __int64 v19; // rdx
  __int64 *v20; // rax
  __int64 v21; // r11
  __int64 *v22; // rdx
  __int64 v23; // rdi
  unsigned int v24; // esi
  unsigned int v25; // ecx
  __int64 *v26; // rax
  unsigned __int64 v28; // [rsp+8h] [rbp-B8h]
  __int64 v29; // [rsp+10h] [rbp-B0h]
  char v30; // [rsp+18h] [rbp-A8h]
  __int64 v32; // [rsp+28h] [rbp-98h]
  __int64 v33; // [rsp+30h] [rbp-90h]
  __int64 v34; // [rsp+38h] [rbp-88h]
  __int64 *v35; // [rsp+40h] [rbp-80h] BYREF
  __int64 v36; // [rsp+48h] [rbp-78h]
  _BYTE v37[112]; // [rsp+50h] [rbp-70h] BYREF

  v3 = *(unsigned int *)(*a1 + 72LL);
  if ( (_DWORD)v3 )
  {
    v30 = 0;
    v5 = 0;
    v33 = 8 * v3;
    while ( 1 )
    {
      while ( 1 )
      {
        v6 = a1[16] + 8 * v5;
        if ( *(_DWORD *)v6 == 1 || !*(_DWORD *)v6 && *(_BYTE *)(v6 + 56) && *(_BYTE *)(v6 + 57) )
        {
          v34 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(*a1 + 64LL) + v5) + 8LL);
          v7 = 0;
          if ( *(_BYTE *)(v6 + 59) )
            v7 = *(_QWORD *)(*(_QWORD *)(v6 + 48) + 8LL);
          v8 = *(_QWORD *)(a2 + 104);
          if ( v8 )
            break;
        }
        v5 += 8;
        if ( v5 == v33 )
        {
LABEL_29:
          if ( v30 )
            sub_2E0AF60(a2);
          return;
        }
      }
      v32 = v5;
      v28 = v7 & 0xFFFFFFFFFFFFFFF8LL;
      v9 = v34 & 0xFFFFFFFFFFFFFFF8LL;
      do
      {
        while ( 1 )
        {
          v12 = (__int64 *)sub_2E09D00((__int64 *)v8, v9);
          v13 = *(_QWORD *)v8 + 24LL * *(unsigned int *)(v8 + 8);
          if ( v12 == (__int64 *)v13 )
            goto LABEL_18;
          v14 = *(_DWORD *)(v9 + 24);
          v15 = *(_DWORD *)((*v12 & 0xFFFFFFFFFFFFFFF8LL) + 24);
          if ( (unsigned __int64)(v15 | (*v12 >> 1) & 3) > v14 )
            break;
          v10 = v12[2];
          if ( v9 == (v12[1] & 0xFFFFFFFFFFFFFFF8LL) )
          {
            if ( (__int64 *)v13 == v12 + 3 )
              goto LABEL_16;
            v15 = *(_DWORD *)((v12[3] & 0xFFFFFFFFFFFFFFF8LL) + 24);
            v12 += 3;
          }
          if ( v9 == *(_QWORD *)(v10 + 8) )
            v10 = 0;
          if ( v14 >= v15 )
            goto LABEL_22;
LABEL_16:
          if ( v10 )
            goto LABEL_17;
LABEL_18:
          v8 = *(_QWORD *)(v8 + 104);
          if ( !v8 )
            goto LABEL_28;
        }
        v10 = 0;
        if ( v14 < v15 )
          goto LABEL_16;
LABEL_22:
        v16 = v12[2];
        v17 = v12[1];
        if ( !v16 )
          goto LABEL_16;
        if ( !v10 )
          goto LABEL_41;
        if ( !*(_BYTE *)(v6 + 59) )
        {
          if ( (((unsigned __int8)v17 ^ 6) & 6) != 0 )
          {
            if ( *(_DWORD *)v6 != 1 )
              goto LABEL_18;
LABEL_33:
            if ( (*(_BYTE *)(v10 + 8) & 6) != 0 || v16 != v10 )
              goto LABEL_18;
          }
LABEL_17:
          v11 = *(_QWORD *)(v8 + 112);
          a3[1] |= *(_QWORD *)(v8 + 120);
          *a3 |= v11;
          goto LABEL_18;
        }
        if ( *(_DWORD *)v6 == 1 )
        {
          if ( *(_QWORD *)(v16 + 8) == v34 )
          {
LABEL_41:
            v18 = a1[7];
            v35 = (__int64 *)v37;
            v29 = v16;
            v36 = 0x800000000LL;
            sub_2E18870(v18, v8, v34, (__int64)&v35);
            *(_QWORD *)(v29 + 8) = 0;
            if ( !*(_BYTE *)(v6 + 59) )
              goto LABEL_42;
            v20 = (__int64 *)sub_2E09D00((__int64 *)v8, v28);
            v21 = v29;
            v22 = v20;
            v23 = *(_QWORD *)v8 + 24LL * *(unsigned int *)(v8 + 8);
            if ( v20 != (__int64 *)v23 )
            {
              v24 = *(_DWORD *)(v28 + 24);
              v25 = *(_DWORD *)((*v20 & 0xFFFFFFFFFFFFFFF8LL) + 24);
              if ( (unsigned __int64)(v25 | (*v20 >> 1) & 3) > v24 || v28 != (v20[1] & 0xFFFFFFFFFFFFFFF8LL) )
                goto LABEL_49;
              v26 = v20 + 3;
              if ( (__int64 *)v23 != v22 + 3 )
              {
                v25 = *(_DWORD *)((v22[3] & 0xFFFFFFFFFFFFFFF8LL) + 24);
                v22 = v26;
LABEL_49:
                if ( v24 >= v25 && v22[2] )
                {
                  sub_2E12C90((_QWORD *)a1[7], v8, v35, (unsigned int)v36, 0, 0);
                  v21 = v29;
                }
              }
            }
            if ( ((*(__int64 *)(v21 + 8) >> 1) & 3) == 0 )
            {
LABEL_42:
              v19 = *(_QWORD *)(v8 + 112);
              a3[1] |= *(_QWORD *)(v8 + 120);
              *a3 |= v19;
            }
            if ( v35 != (__int64 *)v37 )
              _libc_free((unsigned __int64)v35);
            v30 = 1;
            goto LABEL_18;
          }
          if ( (((unsigned __int8)v17 ^ 6) & 6) != 0 )
            goto LABEL_33;
          goto LABEL_17;
        }
        if ( (((unsigned __int8)v17 ^ 6) & 6) == 0 )
          goto LABEL_17;
        v8 = *(_QWORD *)(v8 + 104);
      }
      while ( v8 );
LABEL_28:
      v5 += 8;
      if ( v32 + 8 == v33 )
        goto LABEL_29;
    }
  }
}
