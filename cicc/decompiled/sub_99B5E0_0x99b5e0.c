// Function: sub_99B5E0
// Address: 0x99b5e0
//
__int64 __fastcall sub_99B5E0(__int64 a1, __int64 a2, __int64 a3, __int64 *a4, __int64 a5)
{
  __int64 v5; // r12
  __int64 v7; // rsi
  _QWORD *v8; // rax
  _QWORD *v9; // rdx
  __int64 result; // rax
  __int64 v11; // rdx
  __int64 v12; // rdi
  __int64 v13; // r8
  __int64 v14; // rdx
  unsigned int v15; // esi
  __int64 *v16; // rcx
  __int64 v17; // r10
  __int64 *v18; // r13
  __int64 v19; // rdx
  __int64 v20; // rdi
  __int64 v21; // rax
  __int64 v22; // r12
  __int64 v23; // rcx
  __int64 v24; // rdx
  unsigned int v25; // r13d
  __int64 v26; // r13
  __int64 v27; // rax
  __int64 v28; // rsi
  unsigned int v29; // edx
  __int64 v30; // rcx
  __int64 v31; // rdi
  __int64 v32; // r15
  __int64 v33; // r13
  __int64 v34; // r14
  __int64 v35; // rax
  unsigned __int8 *v36; // r13
  _DWORD *v37; // rax
  int v38; // eax
  unsigned int v39; // ebx
  __int64 v40; // rdx
  unsigned __int64 v41; // r13
  __int64 v42; // rbx
  int v43; // ecx
  int v44; // r9d
  int v45; // ecx
  int v46; // r11d
  unsigned int v48; // [rsp+Ch] [rbp-74h]
  __int64 v49; // [rsp+10h] [rbp-70h]
  __int64 *v50; // [rsp+18h] [rbp-68h]
  __int64 v51; // [rsp+18h] [rbp-68h]
  _QWORD v52[2]; // [rsp+20h] [rbp-60h] BYREF
  __int64 v53; // [rsp+30h] [rbp-50h] BYREF
  unsigned __int64 v54; // [rsp+38h] [rbp-48h]
  __int64 v55; // [rsp+40h] [rbp-40h]

  v5 = a1;
  v7 = a4[7];
  v48 = a3;
  if ( v7 )
  {
    if ( *(_BYTE *)(v7 + 44) )
    {
      v8 = *(_QWORD **)(v7 + 24);
      v9 = &v8[*(unsigned int *)(v7 + 36)];
      if ( v8 == v9 )
        goto LABEL_8;
      while ( a1 != *v8 )
      {
        if ( v9 == ++v8 )
          goto LABEL_8;
      }
    }
    else
    {
      if ( !sub_C8CA60(v7 + 16, a1, a3, a4, a5) )
        goto LABEL_8;
      v7 = a4[7];
    }
    sub_996C70(a1, *(_QWORD *)v7, (_QWORD *)a2, v48, a4, *(_BYTE *)(v7 + 8));
  }
LABEL_8:
  result = a4[5];
  if ( !result )
    return result;
  v11 = a4[6];
  if ( v11 )
  {
    v12 = a4[3];
    if ( v12 )
    {
      v13 = *(_QWORD *)(v11 + 8);
      v14 = *(unsigned int *)(v11 + 24);
      if ( (_DWORD)v14 )
      {
        v15 = (v14 - 1) & (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4));
        v16 = (__int64 *)(v13 + 32LL * v15);
        v17 = *v16;
        if ( v5 == *v16 )
        {
LABEL_13:
          if ( v16 != (__int64 *)(v13 + 32 * v14) )
          {
            v18 = (__int64 *)v16[1];
            v50 = &v18[*((unsigned int *)v16 + 4)];
            if ( v50 != v18 )
            {
              v49 = v5;
              while ( 1 )
              {
                v22 = *v18;
                v23 = *(_QWORD *)(*v18 + 40);
                v52[1] = *(_QWORD *)(*v18 - 32);
                v24 = *(_QWORD *)(result + 40);
                v52[0] = v23;
                if ( (unsigned __int8)sub_B19C20(v12, v52, v24) )
                  sub_996C70(v49, *(_QWORD *)(v22 - 96), (_QWORD *)a2, v48, a4, 0);
                v19 = *(_QWORD *)(v22 + 40);
                v20 = a4[3];
                v54 = *(_QWORD *)(v22 - 64);
                v21 = a4[5];
                v53 = v19;
                if ( (unsigned __int8)sub_B19C20(v20, &v53, *(_QWORD *)(v21 + 40)) )
                  sub_996C70(v49, *(_QWORD *)(v22 - 96), (_QWORD *)a2, v48, a4, 1);
                if ( v50 == ++v18 )
                  break;
                v12 = a4[3];
                result = a4[5];
              }
              v5 = v49;
            }
          }
        }
        else
        {
          v45 = 1;
          while ( v17 != -4096 )
          {
            v46 = v45 + 1;
            v15 = (v14 - 1) & (v45 + v15);
            v16 = (__int64 *)(v13 + 32LL * v15);
            v17 = *v16;
            if ( v5 == *v16 )
              goto LABEL_13;
            v45 = v46;
          }
        }
      }
      v25 = *(_DWORD *)(a2 + 8);
      if ( v25 <= 0x40 )
      {
        result = *(_QWORD *)(a2 + 16) & *(_QWORD *)a2;
        if ( !result )
          goto LABEL_25;
        *(_QWORD *)a2 = 0;
      }
      else
      {
        result = sub_C446A0(a2, a2 + 16);
        if ( !(_BYTE)result )
          goto LABEL_25;
        memset(*(void **)a2, 0, 8 * (((unsigned __int64)v25 + 63) >> 6));
      }
      result = *(unsigned int *)(a2 + 24);
      if ( (unsigned int)result > 0x40 )
        result = (__int64)memset(*(void **)(a2 + 16), 0, 8 * (((unsigned __int64)(unsigned int)result + 63) >> 6));
      else
        *(_QWORD *)(a2 + 16) = 0;
    }
  }
LABEL_25:
  v26 = a4[4];
  if ( !v26 )
    return result;
  if ( *(_BYTE *)(v26 + 192) )
  {
    v27 = *(unsigned int *)(v26 + 184);
    if ( !(_DWORD)v27 )
      goto LABEL_45;
  }
  else
  {
    sub_CFDFC0(a4[4]);
    v27 = *(unsigned int *)(v26 + 184);
    if ( !(_DWORD)v27 )
      goto LABEL_45;
  }
  v28 = *(_QWORD *)(v26 + 168);
  v29 = (v27 - 1) & (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4));
  v30 = v28 + 88LL * v29;
  v31 = *(_QWORD *)(v30 + 24);
  if ( v5 == v31 )
  {
LABEL_29:
    if ( v30 != v28 + 88 * v27 )
    {
      v32 = *(_QWORD *)(v30 + 40);
      v33 = 32LL * *(unsigned int *)(v30 + 48);
      v51 = v32 + v33;
      if ( v32 + v33 != v32 )
      {
        while ( 1 )
        {
          v34 = *(_QWORD *)(v32 + 16);
          if ( !v34 )
            goto LABEL_33;
          v35 = *(unsigned int *)(v32 + 24);
          if ( (_DWORD)v35 != -1 )
          {
            if ( *(_BYTE *)(*(_QWORD *)(v5 + 8) + 8LL) == 14 )
            {
              v40 = 0;
              if ( *(char *)(v34 + 7) < 0 )
              {
                v40 = sub_BD2BC0(*(_QWORD *)(v32 + 16));
                v35 = *(unsigned int *)(v32 + 24);
              }
              sub_CF90E0(&v53, v34, 16 * v35 + v40);
              if ( (_DWORD)v53 == 86 )
              {
                v41 = v54;
                if ( v55 == v5 && v54 && (v54 & (v54 - 1)) == 0 && (unsigned __int8)sub_98CF40(v34, a4[5], a4[3], 1) )
                {
                  _BitScanReverse64(&v41, v41);
                  sub_9870B0(a2, 0, 63 - ((unsigned int)v41 ^ 0x3F));
                }
              }
            }
            goto LABEL_33;
          }
          v36 = *(unsigned __int8 **)(v34 - 32LL * (*(_DWORD *)(v34 + 4) & 0x7FFFFFF));
          if ( (unsigned __int8 *)v5 == v36 && (unsigned __int8)sub_98CF40(*(_QWORD *)(v32 + 16), a4[5], a4[3], 0) )
          {
            sub_987100(a2);
            return sub_986FF0(a2 + 16);
          }
          v54 = v5;
          v53 = 0;
          if ( sub_9987C0((__int64)&v53, 30, v36) && (unsigned __int8)sub_98CF40(v34, a4[5], a4[3], 0) )
          {
            v42 = a2;
            sub_986FF0(a2);
            result = *(unsigned int *)(a2 + 24);
            if ( (unsigned int)result > 0x40 )
              return (__int64)memset(*(void **)(a2 + 16), 0, 8 * (((unsigned __int64)(unsigned int)result + 63) >> 6));
LABEL_65:
            *(_QWORD *)(v42 + 16) = 0;
            return result;
          }
          v37 = (_DWORD *)sub_C94E20(qword_4F862D0);
          v38 = v37 ? *v37 : LODWORD(qword_4F862D0[2]);
          if ( v48 != v38 && *v36 == 82 && (unsigned __int8)sub_98CF40(v34, a4[5], a4[3], 0) )
          {
            v32 += 32;
            sub_993630(v5, (__int64)v36, (_QWORD *)a2, a4, 0);
            if ( v32 == v51 )
              break;
          }
          else
          {
LABEL_33:
            v32 += 32;
            if ( v32 == v51 )
              break;
          }
        }
      }
    }
  }
  else
  {
    v43 = 1;
    while ( v31 != -4096 )
    {
      v44 = v43 + 1;
      v29 = (v27 - 1) & (v43 + v29);
      v30 = v28 + 88LL * v29;
      v31 = *(_QWORD *)(v30 + 24);
      if ( v5 == v31 )
        goto LABEL_29;
      v43 = v44;
    }
  }
LABEL_45:
  v39 = *(_DWORD *)(a2 + 8);
  if ( v39 > 0x40 )
  {
    result = sub_C446A0(a2, a2 + 16);
    if ( !(_BYTE)result )
      return result;
    memset(*(void **)a2, 0, 8 * (((unsigned __int64)v39 + 63) >> 6));
LABEL_64:
    v42 = a2;
    result = *(unsigned int *)(a2 + 24);
    if ( (unsigned int)result > 0x40 )
      return (__int64)memset(*(void **)(a2 + 16), 0, 8 * (((unsigned __int64)(unsigned int)result + 63) >> 6));
    goto LABEL_65;
  }
  result = *(_QWORD *)(a2 + 16) & *(_QWORD *)a2;
  if ( result )
  {
    *(_QWORD *)a2 = 0;
    goto LABEL_64;
  }
  return result;
}
