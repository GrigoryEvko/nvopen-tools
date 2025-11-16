// Function: sub_112BA90
// Address: 0x112ba90
//
void *__fastcall sub_112BA90(__int64 a1, __int64 a2)
{
  __int64 v4; // rdi
  __int64 v5; // rdx
  __int64 v6; // rcx
  __int64 v7; // rdi
  __int64 v8; // rax
  unsigned int v9; // esi
  __int64 *v10; // rdx
  __int64 v11; // r9
  __int64 *v12; // r8
  __int64 *v13; // r14
  __int64 v14; // rbx
  _BYTE *v15; // r9
  __int64 v16; // rdi
  int v17; // eax
  __int64 v18; // rdi
  __int64 v19; // rax
  __int64 v20; // rdx
  __int64 v21; // rax
  __int64 v22; // rdi
  __int64 v23; // rdx
  int v24; // eax
  void *result; // rax
  __int64 v26; // rdx
  _BYTE *v27; // rax
  __int64 v28; // rdx
  _BYTE *v29; // rax
  int v30; // edx
  int v31; // r10d
  __int64 v32; // [rsp+8h] [rbp-D8h]
  __int64 *v33; // [rsp+10h] [rbp-D0h]
  int v34; // [rsp+18h] [rbp-C8h]
  void *v35; // [rsp+18h] [rbp-C8h]
  void *v36; // [rsp+18h] [rbp-C8h]
  _BYTE *v37; // [rsp+18h] [rbp-C8h]
  int v38; // [rsp+2Ch] [rbp-B4h] BYREF
  __int64 v39; // [rsp+30h] [rbp-B0h] BYREF
  _BYTE *v40; // [rsp+38h] [rbp-A8h] BYREF
  __int64 v41[2]; // [rsp+40h] [rbp-A0h] BYREF
  __int64 v42[2]; // [rsp+50h] [rbp-90h] BYREF
  __int64 v43; // [rsp+60h] [rbp-80h] BYREF
  unsigned int v44; // [rsp+68h] [rbp-78h]
  __int64 v45; // [rsp+70h] [rbp-70h]
  unsigned int v46; // [rsp+78h] [rbp-68h]
  _QWORD v47[12]; // [rsp+80h] [rbp-60h] BYREF

  v4 = *(_QWORD *)(a2 - 32);
  v39 = *(_QWORD *)(a2 - 64);
  if ( *(_BYTE *)v4 == 17 )
  {
    v5 = v4 + 24;
    v40 = (_BYTE *)(v4 + 24);
  }
  else
  {
    v28 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v4 + 8) + 8LL) - 17;
    if ( (unsigned int)v28 > 1 )
      return 0;
    if ( *(_BYTE *)v4 > 0x15u )
      return 0;
    v29 = sub_AD7630(v4, 0, v28);
    if ( !v29 || *v29 != 17 )
      return 0;
    v5 = (__int64)(v29 + 24);
    v40 = v29 + 24;
  }
  v38 = *(_WORD *)(a2 + 2) & 0x3F;
  sub_AB1A50((__int64)&v43, v38, v5);
  v6 = *(unsigned int *)(a1 + 224);
  v47[0] = &v43;
  v47[3] = &v38;
  v7 = *(_QWORD *)(a1 + 208);
  v47[4] = &v40;
  v47[5] = &v39;
  v8 = v39;
  v47[1] = a2;
  v47[2] = a1;
  if ( (_DWORD)v6 )
  {
    v9 = (v6 - 1) & (((unsigned int)v39 >> 9) ^ ((unsigned int)v39 >> 4));
    v10 = (__int64 *)(v7 + 32LL * v9);
    v11 = *v10;
    if ( v39 == *v10 )
    {
LABEL_5:
      if ( v10 != (__int64 *)(v7 + 32 * v6) )
      {
        v12 = (__int64 *)v10[1];
        v33 = &v12[*((unsigned int *)v10 + 4)];
        if ( v33 != v12 )
        {
          v13 = (__int64 *)v10[1];
          while ( 1 )
          {
            v14 = *v13;
            v15 = *(_BYTE **)(*v13 - 96);
            if ( *v15 != 82 || *((_QWORD *)v15 - 8) != v8 )
              goto LABEL_8;
            v16 = *((_QWORD *)v15 - 4);
            if ( *(_BYTE *)v16 == 17 )
            {
              v32 = v16 + 24;
            }
            else
            {
              v37 = *(_BYTE **)(*v13 - 96);
              v26 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v16 + 8) + 8LL) - 17;
              if ( (unsigned int)v26 > 1 )
                goto LABEL_8;
              if ( *(_BYTE *)v16 > 0x15u )
                goto LABEL_8;
              v27 = sub_AD7630(v16, 0, v26);
              if ( !v27 || *v27 != 17 )
                goto LABEL_8;
              v15 = v37;
              v32 = (__int64)(v27 + 24);
            }
            v17 = sub_B53900((__int64)v15);
            v18 = *(_QWORD *)(a1 + 80);
            v34 = v17;
            v19 = *(_QWORD *)(v14 - 32);
            v41[0] = *(_QWORD *)(v14 + 40);
            v20 = *(_QWORD *)(a2 + 40);
            v41[1] = v19;
            if ( (unsigned __int8)sub_B19C20(v18, v41, v20) )
              break;
            v21 = *(_QWORD *)(v14 - 64);
            v22 = *(_QWORD *)(a1 + 80);
            v42[0] = *(_QWORD *)(v14 + 40);
            v23 = *(_QWORD *)(a2 + 40);
            v42[1] = v21;
            if ( !(unsigned __int8)sub_B19C20(v22, v42, v23) )
              goto LABEL_8;
            v24 = sub_B52870(v34);
            result = sub_112B2E0((__int64)v47, v24, v32);
            if ( result )
              goto LABEL_19;
            if ( v33 == ++v13 )
              goto LABEL_18;
LABEL_9:
            v8 = v39;
          }
          result = sub_112B2E0((__int64)v47, v34, v32);
          if ( result )
            goto LABEL_19;
LABEL_8:
          if ( v33 == ++v13 )
            goto LABEL_18;
          goto LABEL_9;
        }
      }
    }
    else
    {
      v30 = 1;
      while ( v11 != -4096 )
      {
        v31 = v30 + 1;
        v9 = (v6 - 1) & (v30 + v9);
        v10 = (__int64 *)(v7 + 32LL * v9);
        v11 = *v10;
        if ( v39 == *v10 )
          goto LABEL_5;
        v30 = v31;
      }
    }
  }
LABEL_18:
  result = 0;
LABEL_19:
  if ( v46 > 0x40 && v45 )
  {
    v35 = result;
    j_j___libc_free_0_0(v45);
    result = v35;
  }
  if ( v44 > 0x40 )
  {
    if ( v43 )
    {
      v36 = result;
      j_j___libc_free_0_0(v43);
      return v36;
    }
  }
  return result;
}
