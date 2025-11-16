// Function: sub_1B6FD70
// Address: 0x1b6fd70
//
__int64 __fastcall sub_1B6FD70(__int64 *a1, unsigned __int16 *a2, __int64 a3, _BYTE *a4)
{
  __int64 v5; // rax
  __int64 v6; // rdx
  _QWORD *v7; // r15
  __int64 v9; // rdx
  __int64 v10; // rax
  __int64 ***v11; // r15
  __int64 v12; // rax
  unsigned __int16 *v13; // rax
  unsigned int v14; // r14d
  __int64 **v15; // rax
  unsigned int v16; // ebx
  _QWORD *v17; // rax
  __int64 **v18; // rax
  __int64 v19; // r15
  _QWORD *v20; // rax
  __int64 v21; // rax
  _QWORD *v22; // rax
  __int64 v23; // rax
  __int64 ***v24; // r15
  __int64 **v25; // rax
  __int64 v26; // rax
  unsigned __int16 *v28; // rax
  unsigned __int64 v29; // [rsp+8h] [rbp-48h]
  __int64 *v30; // [rsp+8h] [rbp-48h]
  __int64 *v31; // [rsp+10h] [rbp-40h] BYREF
  _BYTE v32[56]; // [rsp+18h] [rbp-38h] BYREF

  v5 = *(_DWORD *)(a3 + 20) & 0xFFFFFFF;
  v6 = *(_QWORD *)(a3 + 24 * (2 - v5));
  if ( *(_BYTE *)(v6 + 16) != 13 )
    return (unsigned int)-1;
  v7 = *(_QWORD **)(v6 + 24);
  if ( *(_DWORD *)(v6 + 32) > 0x40u )
    v7 = (_QWORD *)*v7;
  v9 = *(_QWORD *)(a3 - 24);
  if ( *(_BYTE *)(v9 + 16) )
    BUG();
  v29 = 8LL * (_QWORD)v7;
  if ( *(_DWORD *)(v9 + 36) != 137 )
  {
    v10 = sub_1649C60(*(_QWORD *)(a3 + 24 * (1 - v5)));
    v11 = (__int64 ***)v10;
    if ( *(_BYTE *)(v10 + 16) <= 0x10u )
    {
      v12 = sub_14AD280(v10, (unsigned __int64)a4, 6u);
      if ( *(_BYTE *)(v12 + 16) == 3 && (*(_BYTE *)(v12 + 80) & 1) != 0 )
      {
        v13 = (unsigned __int16 *)sub_1649C60(*(_QWORD *)(a3 - 24LL * (*(_DWORD *)(a3 + 20) & 0xFFFFFFF)));
        v14 = sub_1B6E190((__int64)a1, a2, v13, v29, (__int64)a4);
        if ( v14 != -1 )
        {
          v15 = *v11;
          if ( *((_BYTE *)*v11 + 8) == 16 )
            v15 = (__int64 **)*v15[2];
          v16 = *((_DWORD *)v15 + 2);
          v17 = (_QWORD *)sub_16498A0((__int64)v11);
          v16 >>= 8;
          v18 = (__int64 **)sub_16471D0(v17, v16);
          v19 = sub_15A4510(v11, v18, 0);
          v20 = (_QWORD *)sub_16498A0(v19);
          v21 = sub_1643360(v20);
          v30 = (__int64 *)sub_159C470(v21, v14, 0);
          v22 = (_QWORD *)sub_16498A0(v19);
          v23 = sub_1643330(v22);
          v31 = v30;
          v32[4] = 0;
          v24 = (__int64 ***)sub_15A2E80(v23, v19, &v31, 1u, 0, (__int64)v32, 0);
          v25 = (__int64 **)sub_1646BA0(a1, v16);
          v26 = sub_15A4510(v24, v25, 0);
          if ( sub_14D8290(v26, (__int64)a1, a4) )
            return v14;
        }
      }
    }
    return (unsigned int)-1;
  }
  v28 = (unsigned __int16 *)sub_1649C60(*(_QWORD *)(a3 - 24 * v5));
  return sub_1B6E190((__int64)a1, a2, v28, v29, (__int64)a4);
}
