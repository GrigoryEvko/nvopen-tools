// Function: sub_921F50
// Address: 0x921f50
//
__int64 __fastcall sub_921F50(__int64 a1, __int64 a2, __int64 *a3, unsigned __int8 a4)
{
  bool v8; // al
  __int64 v9; // rcx
  __int64 v10; // rdi
  unsigned __int8 v11; // r9
  unsigned int v12; // r8d
  unsigned __int64 v14; // rsi
  unsigned int v15; // eax
  __int64 v16; // [rsp+0h] [rbp-60h]
  const char *v17; // [rsp+10h] [rbp-50h] BYREF
  char v18; // [rsp+30h] [rbp-30h]
  char v19; // [rsp+31h] [rbp-2Fh]

  v8 = sub_91B770(*a3);
  v9 = 0;
  if ( v8 )
  {
    v14 = *a3;
    v19 = 1;
    v18 = 3;
    v17 = "agg.tmp";
    v9 = sub_921D70(a2, v14, (__int64)&v17, 0);
  }
  v10 = *a3;
  v11 = a4;
  if ( *(char *)(*a3 + 142) >= 0 && *(_BYTE *)(v10 + 140) == 12 )
  {
    v16 = v9;
    v15 = sub_8D4AB0(v10);
    v9 = v16;
    v11 = a4;
    v12 = v15;
  }
  else
  {
    v12 = *(_DWORD *)(v10 + 136);
  }
  sub_921EA0(a1, a2, a3, v9, v12, v11);
  return a1;
}
