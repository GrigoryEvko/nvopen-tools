// Function: sub_E1ACA0
// Address: 0xe1aca0
//
__int64 __fastcall sub_E1ACA0(__int64 *a1)
{
  __int64 v1; // rax
  _BYTE *v2; // rdx
  __int64 result; // rax
  char v4; // al
  __int64 v5; // rbx
  __int64 v6; // rdx
  __int64 v7; // rcx
  __int64 v8; // r8
  __int64 v9; // r9
  __int64 v10; // r12
  char v11; // dl
  __int64 v12; // rbx
  __int64 v13; // r12
  __int64 v14; // rdx
  __int64 v15; // rcx
  __int64 v16; // r8
  __int64 v17; // r9
  __int64 v18; // r13
  char v19; // dl
  __int64 v20; // rbx
  __int64 v21; // rdx
  __int64 v22; // rcx
  __int64 v23; // r8
  __int64 v24; // r9
  __int64 v25; // r12
  char v26; // dl

  v1 = a1[1];
  v2 = (_BYTE *)*a1;
  if ( *a1 == v1 || v1 - (_QWORD)v2 == 1 || *v2 != 100 )
    return sub_E18BB0((__int64)a1);
  v4 = v2[1];
  switch ( v4 )
  {
    case 'i':
      *a1 = (__int64)(v2 + 2);
      v5 = sub_E12DE0(a1);
      if ( !v5 )
        return 0;
      v10 = sub_E1ACA0(a1);
      if ( !v10 )
        return 0;
      result = sub_E0E790((__int64)(a1 + 102), 40, v6, v7, v8, v9);
      if ( result )
      {
        v11 = *(_BYTE *)(result + 10);
        *(_QWORD *)(result + 16) = v5;
        *(_WORD *)(result + 8) = 16465;
        *(_QWORD *)(result + 24) = v10;
        *(_BYTE *)(result + 32) = 0;
        *(_BYTE *)(result + 10) = v11 & 0xF0 | 5;
        *(_QWORD *)result = &unk_49E0868;
      }
      break;
    case 'x':
      *a1 = (__int64)(v2 + 2);
      v20 = sub_E18BB0((__int64)a1);
      if ( !v20 )
        return 0;
      v25 = sub_E1ACA0(a1);
      if ( !v25 )
        return 0;
      result = sub_E0E790((__int64)(a1 + 102), 40, v21, v22, v23, v24);
      if ( result )
      {
        v26 = *(_BYTE *)(result + 10);
        *(_QWORD *)(result + 16) = v20;
        *(_WORD *)(result + 8) = 16465;
        *(_QWORD *)(result + 24) = v25;
        *(_BYTE *)(result + 32) = 1;
        *(_BYTE *)(result + 10) = v26 & 0xF0 | 5;
        *(_QWORD *)result = &unk_49E0868;
      }
      break;
    case 'X':
      *a1 = (__int64)(v2 + 2);
      v12 = sub_E18BB0((__int64)a1);
      if ( v12 )
      {
        v13 = sub_E18BB0((__int64)a1);
        if ( v13 )
        {
          v18 = sub_E1ACA0(a1);
          if ( v18 )
          {
            result = sub_E0E790((__int64)(a1 + 102), 40, v14, v15, v16, v17);
            if ( result )
            {
              *(_QWORD *)(result + 16) = v12;
              *(_WORD *)(result + 8) = 16466;
              v19 = *(_BYTE *)(result + 10);
              *(_QWORD *)(result + 24) = v13;
              *(_QWORD *)(result + 32) = v18;
              *(_BYTE *)(result + 10) = v19 & 0xF0 | 5;
              *(_QWORD *)result = &unk_49E08C8;
            }
            return result;
          }
        }
      }
      return 0;
    default:
      return sub_E18BB0((__int64)a1);
  }
  return result;
}
