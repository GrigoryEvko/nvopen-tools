// Function: sub_33D2320
// Address: 0x33d2320
//
__int64 __fastcall sub_33D2320(__int64 a1, unsigned int a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 result; // rax
  int v9; // edx
  char v11; // r15
  char v12; // r14
  __int64 v13; // rax
  _BYTE *v14; // r9
  __int64 v15; // r13
  _QWORD *v16; // rsi
  _QWORD *v17; // rax
  _QWORD *v18; // rcx
  __int16 *v19; // rax
  __int64 v20; // r12
  __int16 v21; // r15
  unsigned __int16 v22; // bx
  __int64 v23; // rax
  __int64 v24; // r12
  unsigned __int16 *v25; // rdx
  int v26; // eax
  __int64 v27; // rdx
  __int64 v28; // rdi
  int v29; // edx
  bool v30; // al
  __int64 v31; // rdx
  __int64 v32; // rcx
  __int64 v33; // r8
  unsigned __int16 v34; // ax
  __int64 v35; // rdx
  unsigned __int16 *v36; // rdx
  __int64 v37; // rdx
  __int64 v38; // rdx
  _BYTE *v39; // [rsp+0h] [rbp-B0h]
  __int64 v40; // [rsp+10h] [rbp-A0h]
  unsigned __int16 v41; // [rsp+20h] [rbp-90h] BYREF
  __int64 v42; // [rsp+28h] [rbp-88h]
  _QWORD *v43; // [rsp+30h] [rbp-80h] BYREF
  __int64 v44; // [rsp+38h] [rbp-78h]
  _BYTE v45[48]; // [rsp+40h] [rbp-70h] BYREF
  int v46; // [rsp+70h] [rbp-40h]

  result = a1;
  v9 = *(_DWORD *)(a1 + 24);
  if ( v9 == 11 || v9 == 35 )
    return result;
  v11 = a4;
  v12 = a5;
  if ( v9 == 168 )
  {
    v25 = *(unsigned __int16 **)(a1 + 48);
    v26 = *v25;
    v27 = *((_QWORD *)v25 + 1);
    LOWORD(v43) = v26;
    v44 = v27;
    if ( (_WORD)v26 )
    {
      v28 = 0;
      a4 = (unsigned __int16)word_4456580[v26 - 1];
    }
    else
    {
      a4 = (unsigned int)sub_3009970((__int64)&v43, a3, v27, a4, a5);
      v28 = v38;
    }
    result = **(_QWORD **)(a1 + 40);
    v29 = *(_DWORD *)(result + 24);
    if ( v29 == 35 || v29 == 11 )
    {
      v36 = *(unsigned __int16 **)(result + 48);
      a6 = *v36;
      v37 = *((_QWORD *)v36 + 1);
      if ( v12 || (_WORD)a6 == (_WORD)a4 && ((_WORD)a4 || v37 == v28) )
        return result;
    }
    v9 = *(_DWORD *)(a1 + 24);
  }
  result = 0;
  if ( v9 == 156 )
  {
    v46 = 0;
    v43 = v45;
    v44 = 0x600000000LL;
    v13 = sub_33D22F0(a1, a3, (__int64)&v43, a4, a5, a6);
    v14 = v43;
    v15 = v13;
    if ( !v13 )
      goto LABEL_19;
    v16 = &v43[(unsigned int)v44];
    v17 = sub_33C7FB0(v43, (__int64)v16);
    if ( v18 != v17 && !v11 )
      goto LABEL_19;
    v19 = *(__int16 **)(v15 + 48);
    v20 = *(_QWORD *)(a1 + 48) + 16LL * a2;
    v21 = *v19;
    v22 = *(_WORD *)v20;
    v23 = *((_QWORD *)v19 + 1);
    v24 = *(_QWORD *)(v20 + 8);
    v41 = v22;
    v40 = v23;
    v42 = v24;
    if ( v22 )
    {
      if ( (unsigned __int16)(v22 - 17) <= 0xD3u )
      {
        v24 = 0;
        v22 = word_4456580[v22 - 1];
      }
    }
    else
    {
      v39 = v14;
      v30 = sub_30070B0((__int64)&v41);
      v14 = v39;
      if ( v30 )
      {
        v34 = sub_3009970((__int64)&v41, (__int64)v16, v31, v32, v33);
        v14 = v43;
        v22 = v34;
        v24 = v35;
      }
    }
    if ( v12 || v21 == v22 && (v21 || v40 == v24) )
    {
      if ( v14 != v45 )
        _libc_free((unsigned __int64)v14);
      return v15;
    }
    else
    {
LABEL_19:
      if ( v14 != v45 )
        _libc_free((unsigned __int64)v14);
      return 0;
    }
  }
  return result;
}
