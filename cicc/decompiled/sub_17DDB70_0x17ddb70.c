// Function: sub_17DDB70
// Address: 0x17ddb70
//
__int64 __fastcall sub_17DDB70(__int128 a1, __int64 a2, __int64 a3)
{
  __int64 *v3; // r12
  __int64 v4; // r14
  __int64 v5; // r15
  int v6; // ebx
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 result; // rax
  __int64 v10; // rsi
  __int64 *v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // r14
  __int64 *v15; // rax
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rcx
  __int64 v19; // rax
  __int64 v20; // rax
  const char *v21; // [rsp+0h] [rbp-A0h] BYREF
  char v22; // [rsp+10h] [rbp-90h]
  char v23; // [rsp+11h] [rbp-8Fh]
  __int64 v24[16]; // [rsp+20h] [rbp-80h] BYREF

  v3 = (__int64 *)*((_QWORD *)&a1 + 1);
  v4 = *(_QWORD *)(*((_QWORD *)&a1 + 1) - 24LL);
  v5 = *(_QWORD *)(*((_QWORD *)&a1 + 1) - 48LL);
  if ( *(_BYTE *)(v4 + 16) > 0x10u )
  {
    if ( *(_BYTE *)(v5 + 16) > 0x10u )
      return sub_17DDA20((_QWORD *)a1, *((__int64 *)&a1 + 1));
    v6 = sub_15FF5D0(*(_WORD *)(*((_QWORD *)&a1 + 1) + 18LL) & 0x7FFF);
    v20 = v4;
    v4 = v5;
    v5 = v20;
  }
  else
  {
    v6 = *(unsigned __int16 *)(*((_QWORD *)&a1 + 1) + 18LL);
    BYTE1(v6) &= ~0x80u;
  }
  if ( (!sub_1593BB0(v4, *((__int64 *)&a1 + 1), a2, a3) || (unsigned int)(v6 - 39) > 1)
    && (!sub_1596070(v4, *((__int64 *)&a1 + 1), v7, v8) || v6 != 38 && v6 != 41) )
  {
    return sub_17DDA20((_QWORD *)a1, *((__int64 *)&a1 + 1));
  }
  sub_17CE510((__int64)v24, *((__int64 *)&a1 + 1), 0, 0, 0);
  v23 = 1;
  v21 = "_msprop_icmp_s";
  v22 = 3;
  v10 = *(_QWORD *)v5;
  v11 = sub_17CD8D0((_QWORD *)a1, *(_QWORD *)v5);
  v14 = (__int64)v11;
  if ( v11 )
    v14 = sub_15A06D0((__int64 **)v11, v10, v12, v13);
  *((_QWORD *)&a1 + 1) = v5;
  v15 = sub_17D4DA0(a1);
  v16 = sub_12AA0C0(v24, 0x28u, v15, v14, (__int64)&v21);
  sub_17D4920(a1, v3, v16);
  v19 = sub_17D4880(a1, (const char *)v5, v17, v18);
  result = sub_17D4B80(a1, (__int64)v3, v19);
  if ( v24[0] )
    return sub_161E7C0((__int64)v24, v24[0]);
  return result;
}
