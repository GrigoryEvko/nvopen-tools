// Function: sub_B10E30
// Address: 0xb10e30
//
_QWORD *__fastcall sub_B10E30(_QWORD *a1, __int64 *a2)
{
  _BYTE *v3; // rax
  __int64 v4; // rax
  int v5; // esi
  __int64 v6; // rcx
  __int64 v7; // rax
  __int64 *v8; // rdi
  _QWORD *v9; // rax

  v3 = (_BYTE *)sub_B10DA0(a2);
  v4 = sub_AE7A60(v3);
  if ( v4 )
  {
    v5 = *(_DWORD *)(v4 + 20);
    v6 = v4;
    v7 = *(_QWORD *)(v4 + 8);
    v8 = (__int64 *)(v7 & 0xFFFFFFFFFFFFFFF8LL);
    if ( (v7 & 4) != 0 )
      v8 = (__int64 *)*v8;
    v9 = sub_B01860(v8, v5, 0, v6, 0, 0, 0, 1);
    sub_B10CB0(a1, (__int64)v9);
    return a1;
  }
  else
  {
    *a1 = 0;
    return a1;
  }
}
