// Function: sub_11DD8F0
// Address: 0x11dd8f0
//
unsigned __int64 __fastcall sub_11DD8F0(__int64 a1, __int64 a2, __int64 a3)
{
  int v5; // edx
  __int64 v6; // rdx
  __int64 v7; // r13
  __int64 v8; // r14
  unsigned int v9; // edx
  bool v10; // al
  __int64 v11; // rax
  unsigned int v12; // ebx
  bool v13; // al
  unsigned __int64 result; // rax
  __int64 v15; // rax
  unsigned __int64 v16; // rax
  __int64 v17; // rax
  __int64 v18; // rax
  __int64 *v19; // r15
  __int64 v20; // rax
  _QWORD *v21; // rax
  __int64 v22; // rax
  __int64 *v23; // r15
  __int64 v24; // rax
  __int64 *v25; // rdi
  __int64 **v26; // r13
  unsigned __int64 v27; // rax
  unsigned int v28; // eax
  __int64 v29; // rsi
  _BYTE *v30; // rax
  size_t v31; // r14
  _QWORD *v32; // rdi
  __int64 v33; // rax
  __int64 v34; // [rsp+10h] [rbp-90h]
  __int64 v35; // [rsp+10h] [rbp-90h]
  __int64 v36; // [rsp+18h] [rbp-88h]
  _BYTE *v37; // [rsp+18h] [rbp-88h]
  _BYTE *v38; // [rsp+28h] [rbp-78h] BYREF
  void *s; // [rsp+30h] [rbp-70h] BYREF
  size_t n; // [rsp+38h] [rbp-68h]
  unsigned int v41[8]; // [rsp+40h] [rbp-60h] BYREF
  __int16 v42; // [rsp+60h] [rbp-40h]

  v5 = *(_DWORD *)(a2 + 4);
  v41[0] = 0;
  v6 = v5 & 0x7FFFFFF;
  v7 = *(_QWORD *)(a2 - 32 * v6);
  v8 = *(_QWORD *)(a2 + 32 * (1 - v6));
  sub_11DA4B0(a2, (int *)v41, 1);
  if ( (unsigned __int8)sub_11D9DE0(*(_QWORD *)(a2 + 16), v7) )
    return sub_11DD0A0(a2, 0, (unsigned int **)a3);
  if ( *(_BYTE *)v8 != 17 )
  {
    v16 = sub_98B430(v7, 8u);
    if ( v16 )
    {
      v35 = v16;
      v41[0] = 0;
      sub_11DA2E0(a2, v41, 1, v16);
      v17 = *(_QWORD *)(a2 - 32);
      if ( !v17 || *(_BYTE *)v17 || (v18 = *(_QWORD *)(v17 + 24), v18 != *(_QWORD *)(a2 + 80)) )
        BUG();
      if ( sub_BCAC40(*(_QWORD *)(*(_QWORD *)(v18 + 16) + 16LL), *(_DWORD *)(**(_QWORD **)(a1 + 24) + 172LL)) )
      {
        v19 = *(__int64 **)(a1 + 24);
        v20 = sub_B43CA0(a2);
        LODWORD(v19) = sub_97FA80(*v19, v20);
        v21 = (_QWORD *)sub_BD5C60(a2);
        v22 = sub_BCCE00(v21, (unsigned int)v19);
        v23 = *(__int64 **)(a1 + 24);
        v36 = *(_QWORD *)(a1 + 16);
        v24 = sub_AD64C0(v22, v35, 0);
        result = sub_11CA780(v7, v8, v24, a3, v36, v23);
        if ( result )
        {
          if ( *(_BYTE *)result == 85 )
            *(_WORD *)(result + 2) = *(_WORD *)(result + 2) & 0xFFFC | *(_WORD *)(a2 + 2) & 3;
          return result;
        }
      }
    }
    return 0;
  }
  v9 = *(_DWORD *)(v8 + 32);
  v34 = v8 + 24;
  if ( v9 <= 0x40 )
    v10 = *(_QWORD *)(v8 + 24) == 0;
  else
    v10 = v9 == (unsigned int)sub_C444A0(v34);
  if ( v10 )
  {
    v11 = sub_AD6530(*(_QWORD *)(a2 + 8), v7);
    if ( (unsigned __int8)sub_11D9DE0(*(_QWORD *)(a2 + 16), v11) )
    {
      v25 = *(__int64 **)(a3 + 72);
      v26 = *(__int64 ***)(a2 + 8);
      v42 = 257;
      v27 = sub_ACD6D0(v25);
      return sub_11DB4B0((__int64 *)a3, 0x30u, v27, v26, (__int64)v41, 0, (int)s, 0);
    }
  }
  s = 0;
  n = 0;
  if ( (unsigned __int8)sub_98B0F0(v7, &s, 1u) )
  {
    v28 = *(_DWORD *)(v8 + 32);
    v29 = *(_QWORD *)(v8 + 24);
    if ( v28 > 0x40 )
    {
      v29 = *(_QWORD *)v29;
    }
    else
    {
      if ( !v28 )
        goto LABEL_36;
      v29 = v29 << (64 - (unsigned __int8)v28) >> (64 - (unsigned __int8)v28);
    }
    if ( (_BYTE)v29 )
    {
      if ( !n )
        return sub_AD6530(*(_QWORD *)(a2 + 8), v29);
      v29 = (unsigned int)(char)v29;
      v37 = s;
      v30 = memchr(s, v29, n);
      if ( !v30 )
        return sub_AD6530(*(_QWORD *)(a2 + 8), v29);
      v31 = v30 - v37;
LABEL_37:
      if ( v31 != -1 )
      {
        v32 = *(_QWORD **)(a3 + 72);
        *(_QWORD *)v41 = "strchr";
        v42 = 259;
        v33 = sub_BCB2E0(v32);
        v38 = (_BYTE *)sub_ACD640(v33, v31, 0);
        goto LABEL_13;
      }
      return sub_AD6530(*(_QWORD *)(a2 + 8), v29);
    }
LABEL_36:
    v31 = n;
    goto LABEL_37;
  }
  v12 = *(_DWORD *)(v8 + 32);
  if ( v12 <= 0x40 )
    v13 = *(_QWORD *)(v8 + 24) == 0;
  else
    v13 = v12 == (unsigned int)sub_C444A0(v34);
  if ( !v13 )
    return 0;
  result = sub_11CA050(v7, a3, *(_QWORD *)(a1 + 16), *(__int64 **)(a1 + 24));
  v38 = (_BYTE *)result;
  if ( result )
  {
    *(_QWORD *)v41 = "strchr";
    v42 = 259;
LABEL_13:
    v15 = sub_BCB2B0(*(_QWORD **)(a3 + 72));
    return sub_921130((unsigned int **)a3, v15, v7, &v38, 1, (__int64)v41, 3u);
  }
  return result;
}
