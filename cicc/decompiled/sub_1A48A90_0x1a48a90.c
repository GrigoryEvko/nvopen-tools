// Function: sub_1A48A90
// Address: 0x1a48a90
//
__int64 __fastcall sub_1A48A90(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 ****v4; // rax
  __int64 v6; // r14
  __int64 ***v7; // r13
  __int64 ***v8; // r12
  __int64 v9; // r10
  __int64 *v10; // r15
  int v11; // eax
  int v12; // edi
  __int64 v13; // r8
  __int64 v15; // r8
  __int64 v16; // [rsp-68h] [rbp-68h]
  int v17; // [rsp-60h] [rbp-60h]
  _BYTE v18[16]; // [rsp-58h] [rbp-58h] BYREF
  __int16 v19; // [rsp-48h] [rbp-48h]

  v4 = *(__int64 *****)a1;
  if ( (_DWORD)a2 )
  {
    v6 = (__int64)v4[(unsigned int)a2];
    v7 = v4[(unsigned int)(a2 - 1)];
    v8 = *(__int64 ****)(v6 - 48);
    v9 = sub_1A48A90(a1, (unsigned int)(a2 - 1));
    v10 = *(__int64 **)(v6 + 24LL * (v7 == v8) - 48);
    if ( *(_BYTE *)(v9 + 16) != 13 )
      goto LABEL_5;
    if ( *(_DWORD *)(v9 + 32) <= 0x40u )
    {
      if ( *(_QWORD *)(v9 + 24) )
      {
LABEL_5:
        v12 = *(unsigned __int8 *)(v6 + 16) - 24;
        if ( *(_BYTE *)(v6 + 16) == 51 )
          v12 = 11;
        if ( v7 != v8 )
        {
          v13 = *(_QWORD *)(a1 + 224);
          v19 = 257;
          v10 = (__int64 *)sub_15FB440(v12, v10, v9, (__int64)v18, v13);
LABEL_9:
          sub_164B7C0((__int64)v10, v6);
          return (__int64)v10;
        }
LABEL_15:
        v15 = *(_QWORD *)(a1 + 224);
        v19 = 257;
        v10 = (__int64 *)sub_15FB440(v12, (__int64 *)v9, (__int64)v10, (__int64)v18, v15);
        goto LABEL_9;
      }
    }
    else
    {
      v17 = *(_DWORD *)(v9 + 32);
      v16 = v9;
      v11 = sub_16A57B0(v9 + 24);
      v9 = v16;
      if ( v17 != v11 )
        goto LABEL_5;
    }
    if ( *(_BYTE *)(v6 + 16) != 37 )
      return (__int64)v10;
    v12 = 13;
    if ( v7 != v8 )
      return (__int64)v10;
    goto LABEL_15;
  }
  return sub_15A06D0(**v4, a2, a3, a4);
}
