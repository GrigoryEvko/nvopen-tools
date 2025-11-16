// Function: sub_2608380
// Address: 0x2608380
//
void __fastcall sub_2608380(__int64 a1, __int64 **a2, __int64 a3)
{
  unsigned int *v4; // r14
  __int64 *v5; // r12
  char *v6; // rbx
  char *v7; // r12
  __int64 v8; // rsi
  __int64 v9; // rdi
  __int64 *v10; // rax
  int v11; // ebx
  __int64 *v12; // r13
  __int64 v13; // rsi
  unsigned int v14; // r15d
  int v15; // ebx
  __int64 v16; // r10
  bool v17; // cf
  unsigned int v18; // ebx
  __int64 v19; // r11
  __int64 v20; // r9
  int v21; // edx
  unsigned int v22; // r12d
  unsigned int v23; // eax
  unsigned int v24; // r8d
  int *v25; // rcx
  int v26; // edi
  int v27; // ecx
  __int64 v28; // rdx
  __int64 v29; // r14
  __int64 v30; // r12
  __int64 v31; // rcx
  __int64 v32; // rax
  _BYTE *v33; // rsi
  int v35; // [rsp+20h] [rbp-70h]
  unsigned int v36; // [rsp+24h] [rbp-6Ch]
  __int64 *v37; // [rsp+28h] [rbp-68h]
  __int64 v39; // [rsp+38h] [rbp-58h]
  __int64 v40; // [rsp+40h] [rbp-50h] BYREF
  __int64 v41; // [rsp+48h] [rbp-48h]
  char *v42; // [rsp+50h] [rbp-40h]

  v4 = (unsigned int *)a2[1];
  v5 = *a2;
  sub_25FEA90(&v40, v5, 0x86BCA1AF286BCA1BLL * (((char *)v4 - (char *)v5) >> 3));
  if ( v42 )
    sub_25FFB80((unsigned int *)v5, v4, v42, v41);
  else
    sub_26082D0((unsigned int *)v5, v4);
  v6 = v42;
  v7 = &v42[152 * v41];
  if ( v42 != v7 )
  {
    do
    {
      v8 = *((unsigned int *)v6 + 36);
      v9 = *((_QWORD *)v6 + 16);
      v6 += 152;
      sub_C7D6A0(v9, 8 * v8, 4);
      sub_C7D6A0(*((_QWORD *)v6 - 7), 8LL * *((unsigned int *)v6 - 10), 4);
      sub_C7D6A0(*((_QWORD *)v6 - 11), 16LL * *((unsigned int *)v6 - 18), 8);
      sub_C7D6A0(*((_QWORD *)v6 - 15), 16LL * *((unsigned int *)v6 - 26), 8);
    }
    while ( v7 != v6 );
    v7 = v42;
  }
  j_j___libc_free_0((unsigned __int64)v7);
  v10 = *a2;
  v11 = *((_DWORD *)*a2 + 1);
  if ( v11 != 2 || **(_BYTE **)(v10[1] + 16) != 85 || **(_BYTE **)(v10[2] + 16) != 31 )
  {
    v37 = a2[1];
    if ( v37 != v10 )
    {
      v12 = *a2;
      v36 = 0;
      while ( 1 )
      {
        v13 = v12[1];
        v14 = *(_DWORD *)v12;
        v15 = v11 - 1;
        v16 = *(_QWORD *)(*(_QWORD *)(v13 + 16) + 40LL);
        v39 = *(_QWORD *)(v16 + 72);
        v17 = __CFADD__(*(_DWORD *)v12, v15);
        v18 = *(_DWORD *)v12 + v15;
        if ( !v17 )
        {
          v19 = *(_QWORD *)(a1 + 16);
          v20 = *(unsigned int *)(a1 + 32);
          v21 = 37 * v14;
          v22 = v20 - 1;
          v23 = *(_DWORD *)v12;
          do
          {
            if ( (_DWORD)v20 )
            {
              v24 = v21 & v22;
              v25 = (int *)(v19 + 4LL * (v21 & v22));
              v26 = *v25;
              if ( *v25 == v23 )
              {
LABEL_11:
                if ( v25 != (int *)(v19 + 4 * v20) )
                  goto LABEL_24;
              }
              else
              {
                v27 = 1;
                while ( v26 != -1 )
                {
                  v24 = v22 & (v27 + v24);
                  v35 = v27 + 1;
                  v25 = (int *)(v19 + 4LL * v24);
                  v26 = *v25;
                  if ( *v25 == v23 )
                    goto LABEL_11;
                  v27 = v35;
                }
              }
            }
            ++v23;
            v21 += 37;
          }
          while ( v23 <= v18 );
        }
        v28 = *(_QWORD *)(v12[2] + 8);
        if ( v13 != v28 )
          break;
LABEL_26:
        if ( (unsigned __int8)sub_B2D610(v39, 48)
          || (unsigned __int8)sub_B2D620(v39, "nooutline", 9u)
          || (*(_BYTE *)(sub_B43CB0(*(_QWORD *)(v12[1] + 16)) + 32) & 0xF) == 3 && !*(_BYTE *)a1
          || v36 && v14 <= v36 )
        {
          goto LABEL_24;
        }
        v29 = v12[1];
        v30 = *(_QWORD *)(v12[2] + 8);
        if ( v29 != v30 )
        {
          while ( (unsigned __int8)sub_25F86A0(v29) && (unsigned __int8)sub_25FE580(a1 + 408, *(_QWORD *)(v29 + 16)) )
          {
            v29 = *(_QWORD *)(v29 + 8);
            if ( v29 == v30 )
              goto LABEL_34;
          }
          if ( v29 != v30 )
            goto LABEL_24;
        }
LABEL_34:
        v31 = *(_QWORD *)(a1 + 216);
        *(_QWORD *)(a1 + 296) += 304LL;
        v32 = (v31 + 7) & 0xFFFFFFFFFFFFFFF8LL;
        if ( *(_QWORD *)(a1 + 224) >= (unsigned __int64)(v32 + 304) && v31 )
          *(_QWORD *)(a1 + 216) = v32 + 304;
        else
          v32 = sub_9D1E70(a1 + 216, 304, 304, 3);
        *(_QWORD *)v32 = v12;
        *(_QWORD *)(v32 + 24) = 0xFFFFFFFF00000000LL;
        *(_WORD *)(v32 + 128) = 0;
        *(_QWORD *)(v32 + 200) = v32 + 216;
        *(_QWORD *)(v32 + 208) = 0x400000000LL;
        *(_QWORD *)(v32 + 8) = 0;
        *(_QWORD *)(v32 + 16) = 0;
        *(_QWORD *)(v32 + 32) = 0;
        *(_QWORD *)(v32 + 40) = 0;
        *(_QWORD *)(v32 + 48) = 0;
        *(_DWORD *)(v32 + 56) = 0;
        *(_QWORD *)(v32 + 64) = 0;
        *(_QWORD *)(v32 + 72) = 0;
        *(_QWORD *)(v32 + 80) = 0;
        *(_DWORD *)(v32 + 88) = 0;
        *(_QWORD *)(v32 + 96) = 0;
        *(_QWORD *)(v32 + 104) = 0;
        *(_QWORD *)(v32 + 112) = 0;
        *(_DWORD *)(v32 + 120) = 0;
        *(_QWORD *)(v32 + 136) = 0;
        *(_QWORD *)(v32 + 144) = 0;
        *(_QWORD *)(v32 + 152) = 0;
        *(_DWORD *)(v32 + 160) = 0;
        *(_QWORD *)(v32 + 168) = 0;
        *(_QWORD *)(v32 + 176) = 0;
        *(_QWORD *)(v32 + 184) = 0;
        *(_DWORD *)(v32 + 192) = 0;
        *(_QWORD *)(v32 + 232) = 0;
        *(_QWORD *)(v32 + 240) = 0;
        *(_QWORD *)(v32 + 248) = 0;
        *(_WORD *)(v32 + 256) = 0;
        *(_QWORD *)(v32 + 264) = 0;
        *(_QWORD *)(v32 + 272) = 0;
        *(_QWORD *)(v32 + 280) = 0;
        *(_QWORD *)(v32 + 288) = 0;
        *(_QWORD *)(v32 + 296) = a3;
        *(_QWORD *)(v32 + 272) = *(_QWORD *)(*(_QWORD *)(v12[1] + 16) + 40LL);
        *(_QWORD *)(v32 + 280) = *(_QWORD *)(*(_QWORD *)(v12[2] + 16) + 40LL);
        v33 = *(_BYTE **)(a3 + 8);
        v40 = v32;
        if ( v33 == *(_BYTE **)(a3 + 16) )
        {
          sub_25FD6B0(a3, v33, &v40);
        }
        else
        {
          if ( v33 )
          {
            *(_QWORD *)v33 = v32;
            v33 = *(_BYTE **)(a3 + 8);
          }
          *(_QWORD *)(a3 + 8) = v33 + 8;
        }
        v36 = v18;
        v12 += 19;
        if ( v37 == v12 )
          return;
LABEL_25:
        v11 = *((_DWORD *)v12 + 1);
      }
      while ( (*(_WORD *)(v16 + 2) & 0x7FFF) == 0 )
      {
        v13 = *(_QWORD *)(v13 + 8);
        if ( v13 == v28 )
          goto LABEL_26;
        v16 = *(_QWORD *)(*(_QWORD *)(v13 + 16) + 40LL);
      }
LABEL_24:
      v12 += 19;
      if ( v37 == v12 )
        return;
      goto LABEL_25;
    }
  }
}
