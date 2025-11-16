// Function: sub_2292E00
// Address: 0x2292e00
//
__int64 __fastcall sub_2292E00(__int64 a1, _BYTE *a2, _BYTE *a3, __int64 a4, __int64 a5, __int64 a6, __int64 a7)
{
  __int64 v9; // rdi
  unsigned int v10; // eax
  _BYTE *v11; // r8
  unsigned int v12; // r12d
  size_t v14; // rdx
  int v15; // eax
  __int64 v16; // rax
  __int64 v17; // rax
  _QWORD *v18; // rax
  __int64 v19; // rax
  __int64 v20; // rax
  _QWORD *v21; // rax
  _BYTE *v22; // [rsp+0h] [rbp-A0h]
  unsigned __int64 v23; // [rsp+8h] [rbp-98h]
  __int64 v24; // [rsp+10h] [rbp-90h]
  unsigned __int64 v25; // [rsp+10h] [rbp-90h]
  _BYTE *v26; // [rsp+18h] [rbp-88h]
  __int64 v27; // [rsp+18h] [rbp-88h]
  _BYTE *v29; // [rsp+20h] [rbp-80h]
  __int64 v30; // [rsp+20h] [rbp-80h]
  __int64 v31; // [rsp+20h] [rbp-80h]
  __int64 v32; // [rsp+20h] [rbp-80h]
  __int64 v33; // [rsp+20h] [rbp-80h]
  __int64 v35; // [rsp+28h] [rbp-78h]
  __int64 v36; // [rsp+28h] [rbp-78h]
  void *s1; // [rsp+30h] [rbp-70h] BYREF
  __int64 v38; // [rsp+38h] [rbp-68h]
  _BYTE v39[16]; // [rsp+40h] [rbp-60h] BYREF
  void *s2; // [rsp+50h] [rbp-50h] BYREF
  __int64 v41; // [rsp+58h] [rbp-48h]
  _BYTE v42[64]; // [rsp+60h] [rbp-40h] BYREF

  v9 = *(_QWORD *)(a1 + 8);
  s1 = v39;
  v38 = 0x400000000LL;
  s2 = v42;
  v41 = 0x400000000LL;
  v10 = sub_30B8930(v9, a2, a4, a6, &s1);
  if ( !(_BYTE)v10 )
  {
    v11 = s2;
    v12 = v10;
    goto LABEL_3;
  }
  v12 = sub_30B8930(*(_QWORD *)(a1 + 8), a3, a5, a7, &s2);
  if ( !(_BYTE)v12 )
    goto LABEL_21;
  v11 = s2;
  if ( (unsigned int)v38 != (unsigned __int64)(unsigned int)v41
    || (v14 = 4LL * (unsigned int)v38) != 0 && (v29 = s2, v15 = memcmp(s1, s2, v14), v11 = v29, v15) )
  {
    *(_DWORD *)(a6 + 8) = 0;
    v12 = 0;
    *(_DWORD *)(a7 + 8) = 0;
    goto LABEL_3;
  }
  v26 = 0;
  if ( *a2 > 0x1Cu && (unsigned __int8)(*a2 - 61) <= 1u )
    v26 = (_BYTE *)*((_QWORD *)a2 - 4);
  v22 = 0;
  if ( *a3 > 0x1Cu && (unsigned __int8)(*a3 - 61) <= 1u )
    v22 = (_BYTE *)*((_QWORD *)a3 - 4);
  if ( (_BYTE)qword_4FDB4C8 )
  {
    v12 = (unsigned __int8)qword_4FDB4C8;
    goto LABEL_3;
  }
  v23 = *(unsigned int *)(a6 + 8);
  if ( v23 > 1 )
  {
    v35 = 1;
    do
    {
      v31 = *(_QWORD *)(*(_QWORD *)a6 + 8 * v35);
      if ( !(unsigned __int8)sub_228E2E0(a1, v31, v26) )
        goto LABEL_28;
      v16 = sub_D95540(v31);
      if ( *(_BYTE *)(v16 + 8) == 12 )
      {
        v24 = v31;
        v30 = *(_QWORD *)(a1 + 8);
        v17 = sub_ACD640(v16, *((int *)s1 + v35 - 1), 0);
        v18 = sub_DA2570(v30, v17);
        if ( !(unsigned __int8)sub_228E170(a1, v24, (__int64)v18) )
          goto LABEL_28;
      }
    }
    while ( v23 != ++v35 );
  }
  v25 = *(unsigned int *)(a7 + 8);
  if ( v25 <= 1 )
  {
LABEL_21:
    v11 = s2;
    goto LABEL_3;
  }
  v36 = 1;
  while ( 1 )
  {
    v33 = *(_QWORD *)(*(_QWORD *)a7 + 8 * v36);
    if ( !(unsigned __int8)sub_228E2E0(a1, v33, v22) )
      break;
    v19 = sub_D95540(v33);
    if ( *(_BYTE *)(v19 + 8) == 12 )
    {
      v27 = v33;
      v32 = *(_QWORD *)(a1 + 8);
      v20 = sub_ACD640(v19, *((int *)s2 + v36 - 1), 0);
      v21 = sub_DA2570(v32, v20);
      if ( !(unsigned __int8)sub_228E170(a1, v27, (__int64)v21) )
        break;
    }
    if ( v25 == ++v36 )
      goto LABEL_21;
  }
LABEL_28:
  *(_DWORD *)(a6 + 8) = 0;
  v12 = 0;
  v11 = s2;
  *(_DWORD *)(a7 + 8) = 0;
LABEL_3:
  if ( v11 != v42 )
    _libc_free((unsigned __int64)v11);
  if ( s1 != v39 )
    _libc_free((unsigned __int64)s1);
  return v12;
}
