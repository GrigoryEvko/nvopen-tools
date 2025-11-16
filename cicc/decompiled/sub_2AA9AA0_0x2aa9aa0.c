// Function: sub_2AA9AA0
// Address: 0x2aa9aa0
//
void __fastcall sub_2AA9AA0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rax
  __int64 v7; // r8
  __int64 v8; // r9
  unsigned __int8 v9; // dl
  __int64 v10; // r13
  bool v11; // al
  __int64 v12; // r10
  __int64 v13; // r15
  __int64 v14; // r12
  __int64 v15; // r14
  __int64 v16; // r13
  _BYTE *v17; // rbx
  unsigned __int8 v18; // cl
  __int64 *v19; // rcx
  _QWORD *v20; // rax
  unsigned __int64 v21; // rdx
  __int64 v22; // rax
  unsigned __int64 v23; // rdx
  __int64 *v24; // r12
  __int64 v25; // r8
  __int64 v26; // r9
  __int64 v27; // r15
  __int64 v28; // rax
  unsigned __int64 v29; // rdx
  __m128i *v30; // r12
  __int64 v31; // rdx
  __int64 v32; // rcx
  __int64 v33; // r8
  __int64 v34; // r9
  char v35; // [rsp+18h] [rbp-88h]
  __int64 v36; // [rsp+30h] [rbp-70h] BYREF
  __int64 *v37; // [rsp+40h] [rbp-60h] BYREF
  __int64 v38; // [rsp+48h] [rbp-58h]
  _QWORD v39[10]; // [rsp+50h] [rbp-50h] BYREF

  v37 = v39;
  v38 = 0x400000001LL;
  v39[0] = 0;
  v6 = sub_D49300(a1, a2, a3, a4, a5, a6);
  if ( v6 )
  {
    v9 = *(_BYTE *)(v6 - 16);
    v10 = v6;
    v11 = (v9 & 2) != 0;
    v12 = (v9 & 2) != 0 ? *(unsigned int *)(v10 - 24) : (*(_WORD *)(v10 - 16) >> 6) & 0xFu;
    if ( (unsigned int)v12 > 1 )
    {
      v13 = 8;
      v35 = 0;
      v14 = 8 * v12;
      v15 = v10;
      v16 = v10 - 16;
      if ( (v9 & 2) == 0 )
        goto LABEL_18;
      while ( 1 )
      {
        v17 = *(_BYTE **)(*(_QWORD *)(v15 - 32) + v13);
        if ( (unsigned __int8)(*v17 - 5) <= 0x1Fu )
        {
LABEL_7:
          v18 = *(v17 - 16);
          if ( (v18 & 2) != 0 )
            v19 = (__int64 *)*((_QWORD *)v17 - 4);
          else
            v19 = (__int64 *)&v17[-8 * ((v18 >> 2) & 0xF) - 16];
          v35 = 0;
          if ( !*(_BYTE *)*v19 )
          {
            v20 = (_QWORD *)sub_B91420(*v19);
            if ( v21 <= 0x17 )
            {
              v9 = *(_BYTE *)(v15 - 16);
              v11 = (v9 & 2) != 0;
            }
            else if ( *v20 ^ 0x6F6F6C2E6D766C6CLL | v20[1] ^ 0x6C6C6F726E752E70LL || v20[2] != 0x656C62617369642ELL )
            {
              v9 = *(_BYTE *)(v15 - 16);
              v35 = 0;
              v11 = (v9 & 2) != 0;
            }
            else
            {
              v9 = *(_BYTE *)(v15 - 16);
              v35 = 1;
              v11 = (v9 & 2) != 0;
            }
          }
          if ( v11 )
            v17 = *(_BYTE **)(*(_QWORD *)(v15 - 32) + v13);
          else
            v17 = *(_BYTE **)(v16 + v13 - 8LL * ((v9 >> 2) & 0xF));
        }
        v22 = (unsigned int)v38;
        v23 = (unsigned int)v38 + 1LL;
        if ( v23 > HIDWORD(v38) )
          goto LABEL_20;
        while ( 1 )
        {
          v13 += 8;
          v37[v22] = (__int64)v17;
          LODWORD(v38) = v38 + 1;
          if ( v13 == v14 )
          {
            if ( v35 )
              goto LABEL_25;
            goto LABEL_28;
          }
          v9 = *(_BYTE *)(v15 - 16);
          v11 = (v9 & 2) != 0;
          if ( (v9 & 2) != 0 )
            break;
LABEL_18:
          v17 = *(_BYTE **)(v16 + v13 - 8LL * ((v9 >> 2) & 0xF));
          if ( (unsigned __int8)(*v17 - 5) <= 0x1Fu )
            goto LABEL_7;
          v22 = (unsigned int)v38;
          v23 = (unsigned int)v38 + 1LL;
          if ( v23 > HIDWORD(v38) )
          {
LABEL_20:
            sub_C8D5F0((__int64)&v37, v39, v23, 8u, v7, v8);
            v22 = (unsigned int)v38;
          }
        }
      }
    }
  }
LABEL_28:
  v24 = (__int64 *)sub_AA48A0(**(_QWORD **)(a1 + 32));
  v36 = sub_B9B140(v24, "llvm.loop.unroll.runtime.disable", 0x20u);
  v27 = sub_B9C770(v24, &v36, (__int64 *)1, 0, 1);
  v28 = (unsigned int)v38;
  v29 = (unsigned int)v38 + 1LL;
  if ( v29 > HIDWORD(v38) )
  {
    sub_C8D5F0((__int64)&v37, v39, v29, 8u, v25, v26);
    v28 = (unsigned int)v38;
  }
  v37[v28] = v27;
  LODWORD(v38) = v38 + 1;
  v30 = (__m128i *)sub_B9C770(v24, v37, (__int64 *)(unsigned int)v38, 0, 1);
  sub_BA6610(v30, 0, (unsigned __int8 *)v30);
  sub_D49440(a1, (__int64)v30, v31, v32, v33, v34);
LABEL_25:
  if ( v37 != v39 )
    _libc_free((unsigned __int64)v37);
}
