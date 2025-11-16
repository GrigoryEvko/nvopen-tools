// Function: sub_ADCC50
// Address: 0xadcc50
//
__int64 __fastcall sub_ADCC50(
        __int64 a1,
        _BYTE *a2,
        __int64 a3,
        __int64 a4,
        int a5,
        int a6,
        __int64 a7,
        __int64 a8,
        __int64 a9,
        int a10,
        __int64 a11)
{
  int v13; // ebx
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // rcx
  __int64 v18; // r12
  int v19; // edx

  v13 = (int)a2;
  v14 = sub_BCCE00(*(_QWORD *)(a1 + 8), 64);
  v15 = sub_ACD640(v14, a9, 0);
  sub_B98A20(v15, a9, v16, v17);
  if ( a2 && *a2 == 17 )
    v13 = 0;
  v18 = *(_QWORD *)(a1 + 8);
  v19 = 0;
  if ( a4 )
    v19 = sub_B9B140(v18, a3, a4);
  return sub_B05AE0(v18, 13, v19, a5, a6, v13, a11, a7, 0, a8);
}
