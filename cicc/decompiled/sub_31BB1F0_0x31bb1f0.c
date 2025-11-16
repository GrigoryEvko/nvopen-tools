// Function: sub_31BB1F0
// Address: 0x31bb1f0
//
__int64 __fastcall sub_31BB1F0(__int64 a1, __int64 *a2)
{
  __int64 v3; // rax
  __int64 v4; // r14
  __int64 v5; // rbx
  __int64 v6; // r12
  __int64 v7; // rax
  __int64 v8; // rax
  __int64 *v9; // r12
  __int64 *v10; // rdi
  __int64 v11; // rbx
  __int64 v12; // rax

  v3 = sub_31BAED0(a1, *a2);
  v4 = a2[1];
  v5 = v3;
  if ( *(_DWORD *)(v3 + 16) != 1 )
    v5 = 0;
  if ( v4 )
    v4 = sub_318B4B0(a2[1]);
  v6 = sub_318B4B0(*a2);
  if ( v4 != v6 )
  {
    while ( 1 )
    {
      v7 = sub_31BAED0(a1, v6);
      if ( *(_DWORD *)(v7 + 16) != 1 )
        goto LABEL_8;
      *(_QWORD *)(v7 + 40) = v5;
      if ( v5 )
      {
        *(_QWORD *)(v5 + 48) = v7;
        v5 = v7;
LABEL_8:
        v6 = sub_318B4B0(v6);
        if ( v6 == v4 )
          break;
      }
      else
      {
        v5 = v7;
        v6 = sub_318B4B0(v6);
        if ( v6 == v4 )
          break;
      }
    }
  }
  v8 = *(_QWORD *)(a1 + 32);
  if ( v8 )
  {
    v9 = (__int64 *)(a1 + 32);
    v10 = a2;
    if ( !sub_B445A0(*(_QWORD *)(a2[1] + 16), *(_QWORD *)(v8 + 16)) )
    {
      v10 = (__int64 *)(a1 + 32);
      v9 = a2;
    }
    v11 = sub_31B8FB0(v10, a1);
    v12 = sub_31B8EE0(v9, a1);
    if ( v11 && v12 )
    {
      *(_QWORD *)(v11 + 48) = v12;
      *(_QWORD *)(v12 + 40) = v11;
    }
  }
  return sub_31B9490(a1, (__int64 **)a2);
}
