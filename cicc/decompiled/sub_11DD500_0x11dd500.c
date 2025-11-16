// Function: sub_11DD500
// Address: 0x11dd500
//
_BYTE *__fastcall sub_11DD500(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  _BYTE *result; // rax
  _QWORD *v10; // rdi
  __int64 v11; // rax
  _QWORD **v12; // r15
  unsigned int v13; // eax
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 v16; // [rsp+0h] [rbp-80h]
  __int64 *v17; // [rsp+8h] [rbp-78h]
  _BYTE *v18; // [rsp+18h] [rbp-68h] BYREF
  const char *v19; // [rsp+20h] [rbp-60h] BYREF
  char v20; // [rsp+40h] [rbp-40h]
  char v21; // [rsp+41h] [rbp-3Fh]

  result = (_BYTE *)sub_11CA050(a3, a5, *(_QWORD *)(a1 + 16), *(__int64 **)(a1 + 24));
  v18 = result;
  if ( result )
  {
    v10 = *(_QWORD **)(a5 + 72);
    v21 = 1;
    v19 = "endptr";
    v20 = 3;
    v11 = sub_BCB2B0(v10);
    v16 = sub_921130((unsigned int **)a5, v11, a3, &v18, 1, (__int64)&v19, 3u);
    v17 = *(__int64 **)(a1 + 24);
    v12 = (_QWORD **)sub_AA4B30(*(_QWORD *)(a5 + 48));
    v13 = sub_97FA80(*v17, (__int64)v12);
    v14 = sub_BCCE00(*v12, v13);
    v15 = sub_ACD640(v14, a4 + 1, 0);
    sub_B343C0(a5, 0xEEu, v16, 0x100u, a2, 0x100u, v15, 0, 0, 0, 0, 0);
    return (_BYTE *)a3;
  }
  return result;
}
