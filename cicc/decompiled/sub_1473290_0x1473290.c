// Function: sub_1473290
// Address: 0x1473290
//
__int64 __fastcall sub_1473290(__int64 a1, __int64 a2, _QWORD *a3, __int64 a4, unsigned __int8 a5)
{
  __int64 v8; // rax
  __int64 v9; // rcx
  __int64 v10; // r8
  __int64 v11; // rax
  char v13; // al
  _BOOL4 v14; // eax
  unsigned int i; // r15d
  __int64 v16; // rbx
  __int64 v17; // [rsp+8h] [rbp-58h]
  bool v18; // [rsp+10h] [rbp-50h]
  int v19; // [rsp+14h] [rbp-4Ch]
  __int64 v20; // [rsp+20h] [rbp-40h]
  __int64 v21; // [rsp+28h] [rbp-38h]

  v8 = sub_13FCB50((__int64)a3);
  if ( !v8 || !(unsigned __int8)sub_15CC8F0(*(_QWORD *)(a2 + 56), a4, v8, v9, v10) )
    goto LABEL_3;
  v18 = sub_13F9E70((__int64)a3) != 0;
  v17 = sub_157EBA0(a4);
  v13 = *(_BYTE *)(v17 + 16);
  if ( v13 == 26 )
  {
    v14 = sub_1377F70((__int64)(a3 + 7), *(_QWORD *)(v17 - 24));
    sub_1494CF0(a1, a2, (_DWORD)a3, *(_QWORD *)(v17 - 72), !v14, v18, a5);
    return a1;
  }
  if ( v13 == 27 )
  {
    v19 = sub_15F4D60(v17);
    v20 = sub_157EBA0(a4);
    if ( v19 )
    {
      v21 = 0;
      for ( i = 0; i != v19; ++i )
      {
        v16 = sub_15F4DF0(v20, i);
        if ( !sub_1377F70((__int64)(a3 + 7), v16) )
        {
          if ( v21 )
            goto LABEL_3;
          v21 = v16;
        }
      }
      sub_1473060(a1, a2, a3, v17, v21, v18);
    }
    else
    {
      sub_1473060(a1, a2, a3, v17, 0, v18);
    }
  }
  else
  {
LABEL_3:
    v11 = sub_1456E90(a2);
    sub_14573F0(a1, v11);
  }
  return a1;
}
