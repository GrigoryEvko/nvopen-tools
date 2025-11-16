// Function: sub_2A8B8E0
// Address: 0x2a8b8e0
//
__int64 *__fastcall sub_2A8B8E0(__int64 a1, __int64 a2)
{
  _BYTE *v2; // r13
  __int64 v4; // r14
  __int64 *v5; // rdx
  __int64 v6; // rax
  __int64 v7; // r15
  __int64 v8; // rcx
  int v9; // esi
  char v10; // dl
  int v11; // edx
  unsigned __int64 v12; // rax
  _DWORD *v13; // rax
  unsigned int v14; // r12d
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // rdi
  unsigned __int8 *v18; // r10
  _BYTE *v19; // rbx
  __int64 (__fastcall *v20)(__int64, _BYTE *, _BYTE *, unsigned __int8 *); // rax
  __int64 v21; // rax
  __int64 v22; // r12
  _QWORD *v24; // rax
  __int64 v25; // r9
  __int64 v26; // rbx
  __int64 v27; // r13
  __int64 v28; // rdx
  unsigned int v29; // esi
  __int64 v30; // rax
  int v31; // r8d
  unsigned __int8 *v32; // [rsp+0h] [rbp-B0h]
  unsigned __int8 *v33; // [rsp+0h] [rbp-B0h]
  __int64 v34; // [rsp+10h] [rbp-A0h]
  __int64 *v35; // [rsp+18h] [rbp-98h]
  int v36[8]; // [rsp+20h] [rbp-90h] BYREF
  __int16 v37; // [rsp+40h] [rbp-70h]
  _BYTE v38[32]; // [rsp+50h] [rbp-60h] BYREF
  __int16 v39; // [rsp+70h] [rbp-40h]

  v2 = (_BYTE *)a2;
  v4 = *(_QWORD *)(a1 + 8);
  v5 = *(__int64 **)a1;
  v6 = *(_QWORD *)(a2 + 8);
  v7 = v4 + 56;
  if ( **(_QWORD **)a1 != v6 )
  {
    v39 = 257;
    v8 = *v5;
    if ( *v5 != v6 )
    {
      v9 = *(unsigned __int8 *)(v6 + 8);
      v10 = *(_BYTE *)(v6 + 8);
      if ( (unsigned int)(v9 - 17) > 1 )
      {
        if ( (_BYTE)v9 != 14 )
        {
LABEL_6:
          if ( v9 != 17 )
          {
LABEL_7:
            if ( v10 != 12 )
              goto LABEL_11;
            v11 = *(unsigned __int8 *)(v8 + 8);
            if ( (unsigned int)(v11 - 17) <= 1 )
              LOBYTE(v11) = *(_BYTE *)(**(_QWORD **)(v8 + 16) + 8LL);
            if ( (_BYTE)v11 == 14 )
              v12 = sub_2A8B0B0(
                      (__int64 *)(v4 + 56),
                      0x30u,
                      (unsigned __int64)v2,
                      (__int64 **)v8,
                      (__int64)v38,
                      0,
                      v36[0],
                      0);
            else
LABEL_11:
              v12 = sub_2A8B0B0(
                      (__int64 *)(v4 + 56),
                      0x31u,
                      (unsigned __int64)v2,
                      (__int64 **)v8,
                      (__int64)v38,
                      0,
                      v36[0],
                      0);
            goto LABEL_12;
          }
LABEL_25:
          v10 = *(_BYTE *)(**(_QWORD **)(v6 + 16) + 8LL);
          goto LABEL_7;
        }
      }
      else if ( *(_BYTE *)(**(_QWORD **)(v6 + 16) + 8LL) != 14 )
      {
        goto LABEL_5;
      }
      v31 = *(unsigned __int8 *)(v8 + 8);
      if ( (unsigned int)(v31 - 17) <= 1 )
        LOBYTE(v31) = *(_BYTE *)(**(_QWORD **)(v8 + 16) + 8LL);
      if ( (_BYTE)v31 == 12 )
      {
        v12 = sub_2A8B0B0((__int64 *)(v4 + 56), 0x2Fu, (unsigned __int64)v2, (__int64 **)v8, (__int64)v38, 0, v36[0], 0);
LABEL_12:
        v4 = *(_QWORD *)(a1 + 8);
        v2 = (_BYTE *)v12;
        v7 = v4 + 56;
        goto LABEL_13;
      }
LABEL_5:
      if ( v9 == 18 )
        goto LABEL_25;
      goto LABEL_6;
    }
  }
LABEL_13:
  v13 = *(_DWORD **)(a1 + 24);
  v37 = 257;
  v14 = (*v13)++;
  v15 = sub_BCB2D0(*(_QWORD **)(v4 + 128));
  v16 = sub_ACD640(v15, v14, 0);
  v17 = *(_QWORD *)(v4 + 136);
  v18 = (unsigned __int8 *)v16;
  v35 = *(__int64 **)(a1 + 16);
  v19 = (_BYTE *)*v35;
  v20 = *(__int64 (__fastcall **)(__int64, _BYTE *, _BYTE *, unsigned __int8 *))(*(_QWORD *)v17 + 104LL);
  if ( v20 != sub_948040 )
  {
    v33 = v18;
    v30 = v20(v17, v19, v2, v18);
    v18 = v33;
    v22 = v30;
LABEL_18:
    if ( v22 )
      goto LABEL_19;
    goto LABEL_20;
  }
  if ( *v19 <= 0x15u && *v2 <= 0x15u && *v18 <= 0x15u )
  {
    v32 = v18;
    v21 = sub_AD5A90(*v35, v2, v18, 0);
    v18 = v32;
    v22 = v21;
    goto LABEL_18;
  }
LABEL_20:
  v39 = 257;
  v34 = (__int64)v18;
  v24 = sub_BD2C40(72, 3u);
  v25 = 0;
  v22 = (__int64)v24;
  if ( v24 )
    sub_B4DFA0((__int64)v24, (__int64)v19, (__int64)v2, v34, (__int64)v38, 0, 0, 0);
  (*(void (__fastcall **)(_QWORD, __int64, int *, _QWORD, _QWORD, __int64))(**(_QWORD **)(v4 + 144) + 16LL))(
    *(_QWORD *)(v4 + 144),
    v22,
    v36,
    *(_QWORD *)(v7 + 56),
    *(_QWORD *)(v7 + 64),
    v25);
  v26 = *(_QWORD *)(v4 + 56);
  v27 = v26 + 16LL * *(unsigned int *)(v4 + 64);
  while ( v27 != v26 )
  {
    v28 = *(_QWORD *)(v26 + 8);
    v29 = *(_DWORD *)v26;
    v26 += 16;
    sub_B99FD0(v22, v29, v28);
  }
LABEL_19:
  *v35 = v22;
  return v35;
}
