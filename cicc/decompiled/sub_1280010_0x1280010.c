// Function: sub_1280010
// Address: 0x1280010
//
__int64 __fastcall sub_1280010(__int64 a1, _QWORD *a2, __int64 *a3, unsigned __int8 a4)
{
  bool v8; // al
  _QWORD *v9; // rcx
  __int64 v10; // rdi
  unsigned __int8 v11; // r9
  unsigned int v12; // r8d
  unsigned __int64 v14; // rsi
  unsigned int v15; // eax
  _QWORD *v16; // [rsp+0h] [rbp-50h]
  const char *v17; // [rsp+10h] [rbp-40h] BYREF
  char v18; // [rsp+20h] [rbp-30h]
  char v19; // [rsp+21h] [rbp-2Fh]

  v8 = sub_127B420(*a3);
  v9 = 0;
  if ( v8 )
  {
    v14 = *a3;
    v19 = 1;
    v18 = 3;
    v17 = "agg.tmp";
    v9 = sub_127FE40(a2, v14, (__int64)&v17);
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
  sub_127FF60(a1, (__int64)a2, a3, (__int64)v9, v12, v11);
  return a1;
}
