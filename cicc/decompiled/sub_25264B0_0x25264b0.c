// Function: sub_25264B0
// Address: 0x25264b0
//
__int64 __fastcall sub_25264B0(
        __int64 a1,
        unsigned __int8 (__fastcall *a2)(__int64, unsigned __int64, __int64),
        __int64 a3,
        __int64 a4,
        _BYTE *a5)
{
  __int64 v6; // rax
  __int64 v7; // r15
  __int64 v8; // rax
  unsigned __int8 **v9; // r14
  unsigned __int64 v10; // rax
  __int64 v11; // rcx
  unsigned __int64 v12; // rax
  unsigned __int64 v13; // r15
  int v14; // eax
  unsigned int v15; // r12d
  __int64 v17; // [rsp-10h] [rbp-80h]
  __int64 v18; // [rsp+0h] [rbp-70h]
  unsigned __int8 **i; // [rsp+18h] [rbp-58h]
  _QWORD *v21; // [rsp+20h] [rbp-50h]
  unsigned __int64 v23; // [rsp+30h] [rbp-40h] BYREF
  __int64 v24; // [rsp+38h] [rbp-38h]

  v18 = sub_C996C0("checkForAllReadWriteInstructions", 32, 0, 0);
  v6 = (__int64)sub_250CBE0((__int64 *)(a4 + 72), 32);
  if ( !v6 )
  {
LABEL_13:
    v15 = 0;
    goto LABEL_14;
  }
  v7 = v6;
  v24 = 0;
  v23 = v6 & 0xFFFFFFFFFFFFFFFCLL;
  nullsub_1518();
  v21 = (_QWORD *)sub_251BBC0(a1, v23, 0, a4, 2, 0, 1);
  v8 = sub_251B1C0(*(_QWORD *)(a1 + 208), v7);
  v9 = *(unsigned __int8 ***)(v8 + 32);
  for ( i = &v9[*(unsigned int *)(v8 + 40)]; i != v9; ++v9 )
  {
    v13 = (unsigned __int64)*v9;
    v23 = 0;
    v24 = 0;
    v14 = *(unsigned __int8 *)v13;
    if ( (_BYTE)v14
      && ((unsigned __int8)v14 <= 0x1Cu
       || (v10 = (unsigned int)(v14 - 34), (unsigned __int8)v10 > 0x33u)
       || (v11 = 0x8000000000041LL, !_bittest64(&v11, v10))) )
    {
      v12 = v13 & 0xFFFFFFFFFFFFFFFCLL;
    }
    else
    {
      v12 = v13 & 0xFFFFFFFFFFFFFFFCLL | 2;
    }
    v23 = v12;
    nullsub_1518();
    if ( !(unsigned __int8)sub_251C230(a1, (__int64 *)&v23, a4, v21, a5, 0, 1) && !a2(a3, v13, v17) )
      goto LABEL_13;
  }
  v15 = 1;
LABEL_14:
  if ( v18 )
    sub_C9AF60(v18);
  return v15;
}
