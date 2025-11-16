// Function: sub_E00750
// Address: 0xe00750
//
__int64 __fastcall sub_E00750(__int64 a1, unsigned __int64 a2)
{
  __int64 result; // rax
  unsigned __int8 v5; // al
  bool v6; // dl
  unsigned int v7; // r13d
  unsigned __int64 v8; // r14
  __int64 v9; // rsi
  __int64 v10; // r9
  _QWORD *v11; // rax
  __int64 v12; // r13
  __int64 v13; // rsi
  __int64 v14; // rax
  _QWORD *v15; // rax
  __int64 v16; // r8
  __int64 v17; // rdx
  __int64 v18; // r9
  __int64 v19; // rax
  _QWORD *v20; // rax
  __int64 v21; // r8
  __int64 v22; // rdx
  unsigned __int64 v23; // r9
  __int64 v24; // rbx
  unsigned __int8 v25; // dl
  __int64 v26; // rax
  unsigned __int64 v27; // rdx
  __int64 v28; // rbx
  __int64 v29; // rdx
  __int64 v30; // rcx
  __int64 *v31; // rsi
  __int64 *v32; // rdi
  _QWORD *v33; // [rsp-98h] [rbp-98h]
  __int64 v34; // [rsp-80h] [rbp-80h]
  __int64 v35; // [rsp-78h] [rbp-78h]
  _QWORD *v36; // [rsp-78h] [rbp-78h]
  unsigned __int64 v37; // [rsp-70h] [rbp-70h]
  __int64 v38; // [rsp-70h] [rbp-70h]
  __int64 *v39; // [rsp-68h] [rbp-68h] BYREF
  __int64 v40; // [rsp-60h] [rbp-60h]
  _BYTE v41[88]; // [rsp-58h] [rbp-58h] BYREF

  result = a1;
  if ( a2 )
  {
    v39 = (__int64 *)v41;
    v40 = 0x300000000LL;
    v5 = *(_BYTE *)(a1 - 16);
    v6 = (v5 & 2) != 0;
    if ( (v5 & 2) != 0 )
      v7 = *(_DWORD *)(a1 - 24);
    else
      v7 = (*(_WORD *)(a1 - 16) >> 6) & 0xF;
    v37 = v7;
    if ( v7 )
    {
      v8 = 0;
      v34 = a1 - 16;
      while ( 1 )
      {
        if ( v6 )
          v9 = *(_QWORD *)(a1 - 32);
        else
          v9 = v34 - 8LL * ((v5 >> 2) & 0xF);
        v29 = *(_QWORD *)(*(_QWORD *)(v9 + 8 * v8) + 136LL);
        v10 = *(_QWORD *)(*(_QWORD *)(v9 + 8LL * (unsigned int)(v8 + 1)) + 136LL);
        v11 = *(_QWORD **)(v29 + 24);
        if ( *(_DWORD *)(v29 + 32) > 0x40u )
          v11 = (_QWORD *)*v11;
        v12 = *(_QWORD *)(v10 + 24);
        if ( *(_DWORD *)(v10 + 32) > 0x40u )
          v12 = **(_QWORD **)(v10 + 24);
        if ( a2 < (unsigned __int64)v11 + v12 )
          break;
LABEL_21:
        v8 += 3LL;
        if ( v37 <= v8 )
        {
          v31 = v39;
          v37 = (unsigned int)v40;
          goto LABEL_29;
        }
        v5 = *(_BYTE *)(a1 - 16);
        v6 = (v5 & 2) != 0;
      }
      v13 = (__int64)v11 - a2;
      if ( a2 > (unsigned __int64)v11 )
      {
        v12 += v13;
        v13 = 0;
      }
      v35 = v10;
      v14 = sub_AD64C0(*(_QWORD *)(v29 + 8), v13, 0);
      v15 = sub_B98A20(v14, v13);
      v17 = (unsigned int)v40;
      v18 = v35;
      if ( (unsigned __int64)(unsigned int)v40 + 1 > HIDWORD(v40) )
      {
        v33 = v15;
        sub_C8D5F0((__int64)&v39, v41, (unsigned int)v40 + 1LL, 8u, v16, v35);
        v17 = (unsigned int)v40;
        v15 = v33;
        v18 = v35;
      }
      v39[v17] = (__int64)v15;
      LODWORD(v40) = v40 + 1;
      v19 = sub_AD64C0(*(_QWORD *)(v18 + 8), v12, 0);
      v20 = sub_B98A20(v19, v12);
      v22 = (unsigned int)v40;
      v23 = (unsigned int)v40 + 1LL;
      if ( v23 > HIDWORD(v40) )
      {
        v36 = v20;
        sub_C8D5F0((__int64)&v39, v41, (unsigned int)v40 + 1LL, 8u, v21, v23);
        v22 = (unsigned int)v40;
        v20 = v36;
      }
      v24 = (unsigned int)(v8 + 2);
      v39[v22] = (__int64)v20;
      v25 = *(_BYTE *)(a1 - 16);
      v26 = (unsigned int)(v40 + 1);
      LODWORD(v40) = v40 + 1;
      if ( (v25 & 2) != 0 )
      {
        v27 = v26 + 1;
        v28 = *(_QWORD *)(*(_QWORD *)(a1 - 32) + 8 * v24);
        if ( v26 + 1 <= (unsigned __int64)HIDWORD(v40) )
        {
LABEL_20:
          v39[v26] = v28;
          LODWORD(v40) = v40 + 1;
          goto LABEL_21;
        }
      }
      else
      {
        v30 = v34 - 8LL * ((v25 >> 2) & 0xF);
        v27 = v26 + 1;
        v28 = *(_QWORD *)(v30 + 8 * v24);
        if ( v26 + 1 <= (unsigned __int64)HIDWORD(v40) )
          goto LABEL_20;
      }
      sub_C8D5F0((__int64)&v39, v41, v27, 8u, v21, v23);
      v26 = (unsigned int)v40;
      goto LABEL_20;
    }
    v31 = (__int64 *)v41;
LABEL_29:
    v32 = (__int64 *)(*(_QWORD *)(a1 + 8) & 0xFFFFFFFFFFFFFFF8LL);
    if ( (*(_QWORD *)(a1 + 8) & 4) != 0 )
      v32 = (__int64 *)*v32;
    result = sub_B9C770(v32, v31, (__int64 *)v37, 0, 1);
    if ( v39 != (__int64 *)v41 )
    {
      v38 = result;
      _libc_free(v39, v31);
      return v38;
    }
  }
  return result;
}
