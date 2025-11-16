// Function: sub_11CB6D0
// Address: 0x11cb6d0
//
__int64 __fastcall sub_11CB6D0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 *a6)
{
  __int64 v9; // rbx
  __int64 *v10; // r14
  __int64 v11; // rax
  unsigned int v12; // eax
  __int64 *v13; // rbx
  __int64 v14; // rdx
  int v15; // eax
  unsigned __int64 v16; // rax
  unsigned __int8 *v17; // rdx
  unsigned __int8 *v18; // rax
  __int64 v20; // rax
  __int64 v21; // rcx
  unsigned int v22; // edi
  int *v23; // rdx
  int v24; // esi
  int v25; // edx
  int v26; // r9d
  __int64 v27; // [rsp+8h] [rbp-B8h]
  unsigned __int8 *v28; // [rsp+18h] [rbp-A8h]
  __int64 v29; // [rsp+20h] [rbp-A0h]
  __int64 v30; // [rsp+20h] [rbp-A0h]
  unsigned __int64 v31; // [rsp+28h] [rbp-98h]
  _QWORD v34[4]; // [rsp+40h] [rbp-80h] BYREF
  _QWORD v35[2]; // [rsp+60h] [rbp-60h] BYREF
  _QWORD v36[2]; // [rsp+70h] [rbp-50h] BYREF
  __int64 *v37; // [rsp+80h] [rbp-40h]
  __int64 v38; // [rsp+88h] [rbp-38h]

  v9 = 0;
  v10 = (__int64 *)sub_AA4B30(*(_QWORD *)(a4 + 48));
  if ( !sub_11C99B0(v10, a6, 0x133u) )
    return v9;
  v11 = sub_AA4B30(*(_QWORD *)(a4 + 48));
  v12 = sub_97FA80(*a6, v11);
  v13 = (__int64 *)sub_BCD140(*(_QWORD **)(a4 + 72), v12);
  v31 = a6[5] & 0x8000000000000LL;
  if ( v31 )
  {
    v31 = 0;
    v27 = 0;
    goto LABEL_7;
  }
  v14 = *a6;
  v15 = (int)*(unsigned __int8 *)(*a6 + 76) >> 6;
  if ( !v15 )
  {
    v27 = 0;
    goto LABEL_7;
  }
  if ( v15 != 3 )
  {
    v20 = *(unsigned int *)(v14 + 160);
    v21 = *(_QWORD *)(v14 + 144);
    if ( (_DWORD)v20 )
    {
      v22 = ((_WORD)v20 - 1) & 0x2C5F;
      v23 = (int *)(v21 + 40LL * (((_WORD)v20 - 1) & 0x2C5F));
      v24 = *v23;
      if ( *v23 == 307 )
      {
LABEL_14:
        v27 = *((_QWORD *)v23 + 1);
        v31 = *((_QWORD *)v23 + 2);
        goto LABEL_7;
      }
      v25 = 1;
      while ( v24 != -1 )
      {
        v26 = v25 + 1;
        v22 = (v20 - 1) & (v25 + v22);
        v23 = (int *)(v21 + 40LL * v22);
        v24 = *v23;
        if ( *v23 == 307 )
          goto LABEL_14;
        v25 = v26;
      }
    }
    v23 = (int *)(v21 + 40 * v20);
    goto LABEL_14;
  }
  v27 = 61207541;
  v31 = qword_4977328[614];
LABEL_7:
  v29 = *(_QWORD *)(a3 + 8);
  v36[0] = sub_BCE3C0(*(__int64 **)(a4 + 72), 0);
  v35[0] = v36;
  v38 = v29;
  v36[1] = v13;
  v37 = v13;
  v35[1] = 0x400000004LL;
  v16 = sub_BCF480(v13, v36, 4, 0);
  v30 = sub_11C96C0((__int64)v10, a6, 0x133u, v16, 0);
  v28 = v17;
  if ( *(_BYTE *)(*(_QWORD *)(a3 + 8) + 8LL) == 14 )
    sub_11C9500((__int64)v10, v27, v31, a6);
  LOWORD(v37) = 257;
  v34[0] = a1;
  v34[1] = a2;
  v34[2] = sub_AD64C0((__int64)v13, 1, 0);
  v34[3] = a3;
  v9 = sub_921880((unsigned int **)a4, v30, (int)v28, (int)v34, 4, (__int64)v35, 0);
  v18 = sub_BD3990(v28, v30);
  if ( !*v18 )
    *(_WORD *)(v9 + 2) = *(_WORD *)(v9 + 2) & 0xF003 | (4 * ((*((_WORD *)v18 + 1) >> 4) & 0x3FF));
  return v9;
}
