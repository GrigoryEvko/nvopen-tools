// Function: sub_11CCCD0
// Address: 0x11cccd0
//
__int64 __fastcall sub_11CCCD0(__int64 a1, __int64 a2, __int64 a3, __int64 *a4)
{
  __int64 v4; // r15
  __int64 *v7; // r14
  unsigned __int64 v8; // r15
  __int64 v9; // rdx
  __int64 v10; // rbx
  __int64 v11; // rax
  unsigned int v12; // eax
  __int64 *v13; // rax
  __int64 v14; // rax
  unsigned __int8 *v15; // rdx
  unsigned __int8 *v16; // rax
  __int64 v18; // rax
  __int64 v19; // rcx
  unsigned int v20; // edi
  int *v21; // rdx
  int v22; // esi
  int v23; // edx
  int v24; // r9d
  unsigned __int64 v25; // [rsp+8h] [rbp-78h]
  __int64 v26; // [rsp+10h] [rbp-70h]
  unsigned __int8 *v27; // [rsp+10h] [rbp-70h]
  __int64 v28; // [rsp+18h] [rbp-68h] BYREF
  _QWORD v29[4]; // [rsp+20h] [rbp-60h] BYREF
  __int16 v30; // [rsp+40h] [rbp-40h]

  v4 = 0;
  v28 = a1;
  v7 = (__int64 *)sub_AA4B30(*(_QWORD *)(a2 + 48));
  if ( !sub_11C99B0(v7, a4, 0x161u) )
    return v4;
  v8 = a4[6] & 0x200000000LL;
  if ( v8 )
  {
    v8 = 0;
    v10 = 0;
    goto LABEL_7;
  }
  v9 = *a4;
  if ( (((int)*(unsigned __int8 *)(*a4 + 88) >> 2) & 3) != 0 )
  {
    if ( (((int)*(unsigned __int8 *)(*a4 + 88) >> 2) & 3) == 3 )
    {
      v10 = 66179272;
      v8 = qword_4977328[706];
      goto LABEL_7;
    }
    v18 = *(unsigned int *)(v9 + 160);
    v19 = *(_QWORD *)(v9 + 144);
    if ( (_DWORD)v18 )
    {
      v20 = ((_WORD)v18 - 1) & 0x3305;
      v21 = (int *)(v19 + 40LL * (((_WORD)v18 - 1) & 0x3305));
      v22 = *v21;
      if ( *v21 == 353 )
      {
LABEL_12:
        v10 = *((_QWORD *)v21 + 1);
        v8 = *((_QWORD *)v21 + 2);
        goto LABEL_7;
      }
      v23 = 1;
      while ( v22 != -1 )
      {
        v24 = v23 + 1;
        v20 = (v18 - 1) & (v23 + v20);
        v21 = (int *)(v19 + 40LL * v20);
        v22 = *v21;
        if ( *v21 == 353 )
          goto LABEL_12;
        v23 = v24;
      }
    }
    v21 = (int *)(v19 + 40 * v18);
    goto LABEL_12;
  }
  v10 = 0;
LABEL_7:
  v11 = sub_AA4B30(*(_QWORD *)(a2 + 48));
  v12 = sub_97FA80(*a4, v11);
  v26 = sub_BCD140(*(_QWORD **)(a2 + 72), v12);
  v13 = (__int64 *)sub_BCE3C0(*(__int64 **)(a2 + 72), 0);
  v14 = sub_11CC840((__int64)v7, a4, 0x161u, 0, v13, v26);
  v27 = v15;
  v25 = v14;
  sub_11C9500((__int64)v7, v10, v8, a4);
  v29[1] = v8;
  v30 = 261;
  v29[0] = v10;
  v4 = sub_921880((unsigned int **)a2, v25, (int)v27, (int)&v28, 1, (__int64)v29, 0);
  v16 = sub_BD3990(v27, v25);
  if ( !*v16 )
    *(_WORD *)(v4 + 2) = *(_WORD *)(v4 + 2) & 0xF003 | (4 * ((*((_WORD *)v16 + 1) >> 4) & 0x3FF));
  return v4;
}
