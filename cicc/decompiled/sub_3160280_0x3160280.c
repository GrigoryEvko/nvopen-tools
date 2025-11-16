// Function: sub_3160280
// Address: 0x3160280
//
__int64 __fastcall sub_3160280(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 *v5; // rdi
  __int64 *v6; // rax
  unsigned __int64 v7; // r14
  __int64 **v8; // rax
  int v9; // eax
  __int64 result; // rax
  __int64 v11; // r9
  __int64 v12; // rdx
  __int64 v13; // [rsp+8h] [rbp-78h]
  __int64 v14; // [rsp+18h] [rbp-68h] BYREF
  _QWORD v15[4]; // [rsp+20h] [rbp-60h] BYREF
  __int16 v16; // [rsp+40h] [rbp-40h]

  v5 = *(__int64 **)(a1 + 72);
  v15[0] = *(_QWORD *)(a2 + 8);
  v6 = (__int64 *)sub_BCE3C0(v5, 0);
  v7 = sub_BCF480(v6, v15, 1, 0);
  v8 = (__int64 **)sub_BCE3C0(*(__int64 **)(a1 + 72), 0);
  v9 = sub_AC9EC0(v8);
  v16 = 257;
  v14 = a2;
  result = sub_921880((unsigned int **)a1, v7, v9, (int)&v14, 1, (__int64)v15, 0);
  v12 = *(unsigned int *)(a3 + 256);
  if ( v12 + 1 > (unsigned __int64)*(unsigned int *)(a3 + 260) )
  {
    v13 = result;
    sub_C8D5F0(a3 + 248, (const void *)(a3 + 264), v12 + 1, 8u, v12 + 1, v11);
    v12 = *(unsigned int *)(a3 + 256);
    result = v13;
  }
  *(_QWORD *)(*(_QWORD *)(a3 + 248) + 8 * v12) = result;
  ++*(_DWORD *)(a3 + 256);
  return result;
}
