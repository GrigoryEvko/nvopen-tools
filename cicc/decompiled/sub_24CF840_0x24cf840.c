// Function: sub_24CF840
// Address: 0x24cf840
//
__int64 __fastcall sub_24CF840(__int64 *a1, __int64 a2)
{
  __int64 v2; // rcx
  __int64 v3; // rax
  __int64 v4; // r12
  _BYTE *v6; // r13
  __int64 v7; // rbx
  char v8; // r14
  unsigned __int8 *v9; // rax
  __int64 v10; // rdx
  unsigned __int8 *v11; // r8
  __int64 v12; // rax
  unsigned int v13; // eax
  __int64 *v14; // rdx
  __int64 v15; // r8
  _DWORD *v16; // rax
  unsigned __int8 v17; // al
  unsigned __int8 *v18; // rdi
  __int64 v19; // r9
  unsigned __int64 v20; // rax
  unsigned __int64 v21; // rcx
  int v22; // edx
  __int64 v23; // rax
  __int64 v24; // rdi
  __int64 v25; // rsi
  __int64 v27; // rax
  size_t v28; // rdx
  _QWORD *v29; // r9
  int v30; // eax
  __int64 v31; // rdi
  __int64 v32; // r14
  int v33; // r11d
  __int64 *v34; // rdx
  unsigned int v35; // edi
  __int64 *v36; // rax
  __int64 v37; // rcx
  __int64 *v38; // rax
  unsigned __int64 v39; // r8
  _QWORD *v40; // rax
  int v41; // edx
  int v42; // r9d
  __int64 v43; // rcx
  int v44; // eax
  __int64 v45; // rdi
  int v46; // r10d
  __int64 v47; // r13
  __int64 *v48; // rdi
  __int64 v49; // rsi
  int v50; // r11d
  __int64 *v51; // r8
  __int64 v52; // [rsp+0h] [rbp-B0h]
  __int64 v53; // [rsp+8h] [rbp-A8h]
  unsigned __int8 *v55; // [rsp+18h] [rbp-98h]
  size_t v56; // [rsp+20h] [rbp-90h]
  __int64 v57; // [rsp+28h] [rbp-88h]
  __int64 v58; // [rsp+30h] [rbp-80h]
  _QWORD *v59; // [rsp+30h] [rbp-80h]
  unsigned __int8 *v60; // [rsp+30h] [rbp-80h]
  unsigned __int64 v61; // [rsp+30h] [rbp-80h]
  __int64 v62; // [rsp+40h] [rbp-70h] BYREF
  __int64 v63; // [rsp+48h] [rbp-68h]
  __int64 v64; // [rsp+50h] [rbp-60h]
  unsigned int v65; // [rsp+58h] [rbp-58h]
  void *s2; // [rsp+60h] [rbp-50h] BYREF
  size_t n; // [rsp+68h] [rbp-48h]
  _QWORD v68[8]; // [rsp+70h] [rbp-40h] BYREF

  v2 = *a1;
  v3 = *((unsigned int *)a1 + 2);
  v62 = 0;
  v4 = v2 + 8 * v3;
  v63 = 0;
  v64 = 0;
  v65 = 0;
  v57 = v2;
  if ( v2 == v4 )
  {
    v24 = 0;
    v25 = 0;
    goto LABEL_27;
  }
  do
  {
    v6 = *(_BYTE **)(v4 - 8);
    v7 = *((_QWORD *)v6 - 4);
    v8 = *v6;
    v58 = sub_B43CA0((__int64)v6);
    v9 = sub_BD4CB0((unsigned __int8 *)v7, (void (__fastcall *)(__int64, unsigned __int8 *))nullsub_96, (__int64)&s2);
    v11 = v9;
    if ( *v9 == 3 && (v9[35] & 4) != 0 )
    {
      v55 = v9;
      v27 = sub_B31D10((__int64)v9, (__int64)nullsub_96, v10);
      v56 = v28;
      v53 = v27;
      sub_ED12E0((__int64)&s2, 1, *(_DWORD *)(v58 + 284), 0);
      v29 = s2;
      v11 = v55;
      if ( n <= v56 )
      {
        if ( !n || (v59 = s2, v30 = memcmp((const void *)(v56 - n + v53), s2, n), v29 = v59, !v30) )
        {
          if ( v29 != v68 )
            j_j___libc_free_0((unsigned __int64)v29);
          goto LABEL_25;
        }
        v11 = v55;
      }
      if ( v29 != v68 )
      {
        v60 = v11;
        j_j___libc_free_0((unsigned __int64)v29);
        v11 = v60;
      }
    }
    v12 = *((_QWORD *)v11 + 1);
    if ( (unsigned int)*(unsigned __int8 *)(v12 + 8) - 17 <= 1 )
    {
      v12 = **(_QWORD **)(v12 + 16);
      if ( (unsigned int)*(unsigned __int8 *)(v12 + 8) - 17 <= 1 )
        v12 = **(_QWORD **)(v12 + 16);
    }
    if ( !(*(_DWORD *)(v12 + 8) >> 8) )
    {
      if ( v8 != 62 )
      {
        if ( v65 )
        {
          v13 = (v65 - 1) & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
          v14 = (__int64 *)(v63 + 16LL * v13);
          v15 = *v14;
          if ( *v14 == v7 )
          {
LABEL_11:
            if ( !(_BYTE)qword_4FEDE48 && v14 != (__int64 *)(16LL * v65 + v63) )
            {
              v16 = (_DWORD *)(*(_QWORD *)a2 + 16 * v14[1]);
              if ( !(_BYTE)qword_4FEDF28 || (v6[2] & 1) == 0 && (*(_BYTE *)(*(_QWORD *)v16 + 2LL) & 1) == 0 )
              {
                v16[2] |= 1u;
                goto LABEL_25;
              }
            }
          }
          else
          {
            v41 = 1;
            while ( v15 != -4096 )
            {
              v42 = v41 + 1;
              v13 = (v65 - 1) & (v41 + v13);
              v14 = (__int64 *)(v63 + 16LL * v13);
              v15 = *v14;
              if ( *v14 == v7 )
                goto LABEL_11;
              v41 = v42;
            }
          }
        }
        v17 = *(_BYTE *)v7;
        v18 = (unsigned __int8 *)v7;
        if ( *(_BYTE *)v7 == 63 )
        {
          v18 = *(unsigned __int8 **)(v7 - 32LL * (*(_DWORD *)(v7 + 4) & 0x7FFFFFF));
          v17 = *v18;
        }
        if ( v17 == 3 )
        {
          if ( (v18[80] & 1) != 0 )
            goto LABEL_25;
        }
        else if ( v17 == 61 && (v18[7] & 0x20) != 0 )
        {
          v31 = sub_B91C10((__int64)v18, 1);
          if ( v31 )
          {
            if ( (unsigned __int8)sub_DFFEB0(v31) )
              goto LABEL_25;
          }
        }
      }
      if ( *sub_98ACB0((unsigned __int8 *)v7, 6u) != 60 )
      {
        v20 = *(unsigned int *)(a2 + 8);
        v21 = *(unsigned int *)(a2 + 12);
        v22 = *(_DWORD *)(a2 + 8);
        if ( v20 >= v21 )
          goto LABEL_45;
        goto LABEL_21;
      }
      if ( (unsigned __int8)sub_D13FA0(v7, 1, 0) )
      {
        v20 = *(unsigned int *)(a2 + 8);
        v21 = *(unsigned int *)(a2 + 12);
        v22 = *(_DWORD *)(a2 + 8);
        if ( v20 >= v21 )
        {
LABEL_45:
          v39 = v52 & 0xFFFFFFFF00000000LL;
          v52 &= 0xFFFFFFFF00000000LL;
          if ( v21 < v20 + 1 )
          {
            v61 = v39;
            sub_C8D5F0(a2, (const void *)(a2 + 16), v20 + 1, 0x10u, v39, v19);
            v20 = *(unsigned int *)(a2 + 8);
            v39 = v61;
          }
          v40 = (_QWORD *)(*(_QWORD *)a2 + 16 * v20);
          *v40 = v6;
          v40[1] = v39;
          ++*(_DWORD *)(a2 + 8);
LABEL_24:
          if ( v8 != 62 )
            goto LABEL_25;
          v32 = *(unsigned int *)(a2 + 8) - 1LL;
          if ( v65 )
          {
            v33 = 1;
            v34 = 0;
            v35 = (v65 - 1) & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
            v36 = (__int64 *)(v63 + 16LL * v35);
            v37 = *v36;
            if ( *v36 == v7 )
            {
LABEL_41:
              v38 = v36 + 1;
LABEL_42:
              *v38 = v32;
              goto LABEL_25;
            }
            while ( v37 != -4096 )
            {
              if ( !v34 && v37 == -8192 )
                v34 = v36;
              v35 = (v65 - 1) & (v33 + v35);
              v36 = (__int64 *)(v63 + 16LL * v35);
              v37 = *v36;
              if ( *v36 == v7 )
                goto LABEL_41;
              ++v33;
            }
            if ( !v34 )
              v34 = v36;
            ++v62;
            v44 = v64 + 1;
            if ( 4 * ((int)v64 + 1) < 3 * v65 )
            {
              if ( v65 - HIDWORD(v64) - v44 <= v65 >> 3 )
              {
                sub_9BBF00((__int64)&v62, v65);
                if ( !v65 )
                {
LABEL_94:
                  LODWORD(v64) = v64 + 1;
                  BUG();
                }
                v46 = 1;
                LODWORD(v47) = (v65 - 1) & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
                v48 = 0;
                v44 = v64 + 1;
                v34 = (__int64 *)(v63 + 16LL * (unsigned int)v47);
                v49 = *v34;
                if ( v7 != *v34 )
                {
                  while ( v49 != -4096 )
                  {
                    if ( v49 == -8192 && !v48 )
                      v48 = v34;
                    v47 = (v65 - 1) & ((_DWORD)v47 + v46);
                    v34 = (__int64 *)(v63 + 16 * v47);
                    v49 = *v34;
                    if ( *v34 == v7 )
                      goto LABEL_60;
                    ++v46;
                  }
                  if ( v48 )
                    v34 = v48;
                }
              }
              goto LABEL_60;
            }
          }
          else
          {
            ++v62;
          }
          sub_9BBF00((__int64)&v62, 2 * v65);
          if ( !v65 )
            goto LABEL_94;
          LODWORD(v43) = (v65 - 1) & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
          v44 = v64 + 1;
          v34 = (__int64 *)(v63 + 16LL * (unsigned int)v43);
          v45 = *v34;
          if ( v7 != *v34 )
          {
            v50 = 1;
            v51 = 0;
            while ( v45 != -4096 )
            {
              if ( v45 == -8192 && !v51 )
                v51 = v34;
              v43 = (v65 - 1) & ((_DWORD)v43 + v50);
              v34 = (__int64 *)(v63 + 16 * v43);
              v45 = *v34;
              if ( *v34 == v7 )
                goto LABEL_60;
              ++v50;
            }
            if ( v51 )
              v34 = v51;
          }
LABEL_60:
          LODWORD(v64) = v44;
          if ( *v34 != -4096 )
            --HIDWORD(v64);
          *v34 = v7;
          v38 = v34 + 1;
          v34[1] = 0;
          goto LABEL_42;
        }
LABEL_21:
        v23 = *(_QWORD *)a2 + 16 * v20;
        if ( v23 )
        {
          *(_QWORD *)v23 = v6;
          *(_DWORD *)(v23 + 8) = 0;
          v22 = *(_DWORD *)(a2 + 8);
        }
        *(_DWORD *)(a2 + 8) = v22 + 1;
        goto LABEL_24;
      }
    }
LABEL_25:
    v4 -= 8;
  }
  while ( v57 != v4 );
  v24 = v63;
  v25 = 16LL * v65;
LABEL_27:
  *((_DWORD *)a1 + 2) = 0;
  return sub_C7D6A0(v24, v25, 8);
}
