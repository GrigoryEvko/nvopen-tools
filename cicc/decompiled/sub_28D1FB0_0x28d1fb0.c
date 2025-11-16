// Function: sub_28D1FB0
// Address: 0x28d1fb0
//
__int64 __fastcall sub_28D1FB0(__int64 a1, __int64 a2)
{
  __int64 v2; // r14
  __int64 v3; // r13
  unsigned int v4; // eax
  _BYTE **v5; // rbx
  _BYTE *v6; // r13
  __int64 *v7; // rax
  _QWORD *v8; // rax
  _QWORD *v9; // rdx
  __int64 *v10; // rax
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // r9
  __int64 result; // rax
  __int64 *v15; // rax
  unsigned int v16; // ecx
  __int64 *v17; // rax
  unsigned int v18; // edx
  __int64 v19; // rax
  unsigned __int64 v20; // rdx
  __int64 v21; // rbx
  __int64 v22; // rcx
  __int64 v23; // rax
  __int64 v24; // rdx
  __int64 v25; // r14
  __int64 v26; // rbx
  __int64 v28; // rsi
  _QWORD *v29; // rax
  __int64 v30; // rsi
  __int64 v31; // rdx
  unsigned int v32; // esi
  __int64 v33; // rdi
  int v34; // ecx
  __int64 v35; // r8
  __int64 v36; // r10
  unsigned int v37; // edx
  _QWORD *v38; // rax
  __int64 v39; // r9
  _DWORD *v40; // rax
  __int64 v41; // rdi
  __int64 v42; // rsi
  int v43; // eax
  unsigned int v44; // r8d
  __int64 *v45; // rdx
  __int64 v46; // r10
  _QWORD *v47; // rax
  __int64 v48; // rdx
  int v49; // edx
  int v50; // eax
  int v51; // edx
  int v52; // ecx
  __int64 v53; // [rsp+8h] [rbp-78h]
  unsigned int v54; // [rsp+10h] [rbp-70h]
  unsigned int v55; // [rsp+14h] [rbp-6Ch]
  _BYTE **v56; // [rsp+18h] [rbp-68h]
  int v57; // [rsp+18h] [rbp-68h]
  __int64 v58; // [rsp+28h] [rbp-58h] BYREF
  _QWORD *v59; // [rsp+30h] [rbp-50h] BYREF
  __int64 v60; // [rsp+38h] [rbp-48h]

  v2 = a1 + 104;
  v3 = a2;
  v4 = *(_DWORD *)a1;
  v59 = (_QWORD *)a2;
  *(_DWORD *)a1 = v4 + 1;
  *(_DWORD *)sub_28D1E70(a1 + 104, (__int64 *)&v59) = v4 + 1;
  v55 = *(_DWORD *)a1;
  if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
  {
    v5 = *(_BYTE ***)(a2 - 8);
    v56 = &v5[4 * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)];
  }
  else
  {
    v56 = (_BYTE **)a2;
    v5 = (_BYTE **)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
  }
  if ( v5 != v56 )
  {
    while ( 1 )
    {
      v6 = *v5;
      if ( **v5 <= 0x1Cu )
        goto LABEL_13;
      v59 = *v5;
      v7 = sub_28CBE90(v2, (__int64 *)&v59);
      if ( !v7 || !*((_DWORD *)v7 + 2) )
      {
        sub_28D1FB0(a1, v6);
        v6 = *v5;
      }
      if ( *(_BYTE *)(a1 + 36) )
      {
        v8 = *(_QWORD **)(a1 + 16);
        v9 = &v8[*(unsigned int *)(a1 + 28)];
        if ( v8 == v9 )
          goto LABEL_24;
        while ( (_BYTE *)*v8 != v6 )
        {
          if ( v9 == ++v8 )
            goto LABEL_24;
        }
LABEL_13:
        v5 += 4;
        if ( v56 == v5 )
          goto LABEL_14;
      }
      else
      {
        if ( sub_C8CA60(a1 + 8, (__int64)v6) )
          goto LABEL_13;
        v6 = *v5;
LABEL_24:
        v59 = v6;
        v15 = sub_28CBE90(v2, (__int64 *)&v59);
        v16 = 0;
        if ( v15 )
          v16 = *((_DWORD *)v15 + 2);
        v54 = v16;
        v59 = (_QWORD *)a2;
        v17 = sub_28CBE90(v2, (__int64 *)&v59);
        v18 = 0;
        if ( v17 )
        {
          v18 = *((_DWORD *)v17 + 2);
          if ( v18 > v54 )
            v18 = v54;
        }
        v5 += 4;
        v59 = (_QWORD *)a2;
        *(_DWORD *)sub_28D1E70(v2, (__int64 *)&v59) = v18;
        if ( v56 == v5 )
        {
LABEL_14:
          v3 = a2;
          break;
        }
      }
    }
  }
  v59 = (_QWORD *)v3;
  v10 = sub_28CBE90(v2, (__int64 *)&v59);
  if ( v10 )
    LODWORD(v10) = *((_DWORD *)v10 + 2);
  if ( v55 != (_DWORD)v10 )
  {
    result = *(unsigned int *)(a1 + 144);
    if ( result + 1 > (unsigned __int64)*(unsigned int *)(a1 + 148) )
    {
      sub_C8D5F0(a1 + 136, (const void *)(a1 + 152), result + 1, 8u, v12, v13);
      result = *(unsigned int *)(a1 + 144);
    }
    *(_QWORD *)(*(_QWORD *)(a1 + 136) + 8 * result) = v3;
    ++*(_DWORD *)(a1 + 144);
    return result;
  }
  v19 = *(unsigned int *)(a1 + 224);
  v20 = *(unsigned int *)(a1 + 228);
  v21 = v19 + 1;
  v57 = *(_DWORD *)(a1 + 224);
  if ( v20 < v19 + 1 )
  {
    sub_28CF940(a1 + 216, v19 + 1, v20, v11, v12, v13);
    v19 = *(unsigned int *)(a1 + 224);
  }
  v22 = *(_QWORD *)(a1 + 216);
  v23 = v22 + 96 * v19;
  v24 = v22 + 96 * v21;
  if ( v23 != v24 )
  {
    do
    {
      if ( v23 )
      {
        *(_QWORD *)v23 = 0;
        *(_QWORD *)(v23 + 8) = v23 + 32;
        *(_DWORD *)(v23 + 16) = 8;
        *(_DWORD *)(v23 + 20) = 0;
        *(_DWORD *)(v23 + 24) = 0;
        *(_BYTE *)(v23 + 28) = 1;
      }
      v23 += 96;
    }
    while ( v24 != v23 );
    v22 = *(_QWORD *)(a1 + 216);
  }
  v25 = a1 + 8;
  *(_DWORD *)(a1 + 224) = v57 + 1;
  v26 = v22 + 96LL * (unsigned int)v21 - 96;
  sub_AE6EC0(v26, v3);
  sub_AE6EC0(a1 + 8, v3);
  v59 = (_QWORD *)v3;
  v53 = a1 + 1000;
  *(_DWORD *)sub_28D1E70(a1 + 1000, (__int64 *)&v59) = v57;
  result = *(unsigned int *)(a1 + 144);
  if ( (_DWORD)result )
  {
    while ( 1 )
    {
      v41 = *(_QWORD *)(a1 + 112);
      v42 = *(_QWORD *)(*(_QWORD *)(a1 + 136) + 8 * result - 8);
      result = *(unsigned int *)(a1 + 128);
      if ( (_DWORD)result )
      {
        v43 = result - 1;
        v44 = v43 & (((unsigned int)v42 >> 9) ^ ((unsigned int)v42 >> 4));
        v45 = (__int64 *)(v41 + 16LL * v44);
        v46 = *v45;
        if ( v42 == *v45 )
        {
LABEL_50:
          result = *((unsigned int *)v45 + 2);
        }
        else
        {
          v49 = 1;
          while ( v46 != -4096 )
          {
            v52 = v49 + 1;
            v44 = v43 & (v49 + v44);
            v45 = (__int64 *)(v41 + 16LL * v44);
            v46 = *v45;
            if ( v42 == *v45 )
              goto LABEL_50;
            v49 = v52;
          }
          result = 0;
        }
      }
      if ( v55 > (unsigned int)result )
        return result;
      v58 = v42;
      v47 = sub_AE6EC0(v26, v42);
      v28 = *(_BYTE *)(v26 + 28) ? *(unsigned int *)(v26 + 20) : *(unsigned int *)(v26 + 16);
      v48 = *(_QWORD *)(v26 + 8) + 8 * v28;
      v59 = v47;
      v60 = v48;
      sub_254BBF0((__int64)&v59);
      v29 = sub_AE6EC0(v25, v58);
      v30 = *(_BYTE *)(a1 + 36) ? *(unsigned int *)(a1 + 28) : *(unsigned int *)(a1 + 24);
      v31 = *(_QWORD *)(a1 + 16) + 8 * v30;
      v59 = v29;
      v60 = v31;
      sub_254BBF0((__int64)&v59);
      v32 = *(_DWORD *)(a1 + 1024);
      if ( !v32 )
        break;
      v33 = v58;
      v34 = 1;
      v35 = 0;
      v36 = *(_QWORD *)(a1 + 1008);
      v37 = (v32 - 1) & (((unsigned int)v58 >> 9) ^ ((unsigned int)v58 >> 4));
      v38 = (_QWORD *)(v36 + 16LL * v37);
      v39 = *v38;
      if ( *v38 != v58 )
      {
        while ( v39 != -4096 )
        {
          if ( !v35 && v39 == -8192 )
            v35 = (__int64)v38;
          v37 = (v32 - 1) & (v34 + v37);
          v38 = (_QWORD *)(v36 + 16LL * v37);
          v39 = *v38;
          if ( v58 == *v38 )
            goto LABEL_46;
          ++v34;
        }
        if ( !v35 )
          v35 = (__int64)v38;
        v50 = *(_DWORD *)(a1 + 1016);
        ++*(_QWORD *)(a1 + 1000);
        v51 = v50 + 1;
        v59 = (_QWORD *)v35;
        if ( 4 * (v50 + 1) < 3 * v32 )
        {
          if ( v32 - *(_DWORD *)(a1 + 1020) - v51 > v32 >> 3 )
          {
LABEL_68:
            *(_DWORD *)(a1 + 1016) = v51;
            if ( *(_QWORD *)v35 != -4096 )
              --*(_DWORD *)(a1 + 1020);
            *(_QWORD *)v35 = v33;
            v40 = (_DWORD *)(v35 + 8);
            *(_DWORD *)(v35 + 8) = 0;
            goto LABEL_47;
          }
LABEL_73:
          sub_A429D0(v53, v32);
          sub_A56BF0(v53, &v58, &v59);
          v33 = v58;
          v35 = (__int64)v59;
          v51 = *(_DWORD *)(a1 + 1016) + 1;
          goto LABEL_68;
        }
LABEL_72:
        v32 *= 2;
        goto LABEL_73;
      }
LABEL_46:
      v40 = v38 + 1;
LABEL_47:
      *v40 = v57;
      result = (unsigned int)(*(_DWORD *)(a1 + 144) - 1);
      *(_DWORD *)(a1 + 144) = result;
      if ( !(_DWORD)result )
        return result;
    }
    ++*(_QWORD *)(a1 + 1000);
    v59 = 0;
    goto LABEL_72;
  }
  return result;
}
