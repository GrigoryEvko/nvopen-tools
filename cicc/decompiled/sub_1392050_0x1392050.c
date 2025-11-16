// Function: sub_1392050
// Address: 0x1392050
//
__int64 __fastcall sub_1392050(__int64 a1, __int64 a2)
{
  __int64 v3; // r13
  __int64 v4; // r14
  __int64 result; // rax
  unsigned __int8 v6; // al
  __int64 v7; // r13
  __int64 v8; // rax
  __int64 v9; // rdi
  __int64 v10; // rdx
  unsigned __int8 v11; // al
  __int64 v12; // rbx
  __int64 v13; // rax
  __int64 v14; // rdi
  __int64 v15; // rdx
  __int64 v16; // rbx
  __int64 v17; // rax
  __int64 v18; // rdx
  __int64 v19; // rdx
  __int64 v20; // r13
  __int64 v21; // rax

  v3 = *(_QWORD *)(a2 - 48);
  v4 = *(_QWORD *)(a2 - 24);
  if ( *(_BYTE *)(*(_QWORD *)v3 + 8LL) == 15 )
  {
    result = *(_QWORD *)a2;
    if ( *(_BYTE *)(*(_QWORD *)a2 + 8LL) != 15 )
      return result;
    v11 = *(_BYTE *)(v3 + 16);
    if ( v11 > 3u )
    {
      if ( v11 == 5 )
      {
        if ( (unsigned int)*(unsigned __int16 *)(v3 + 18) - 51 > 1
          && (unsigned __int8)sub_13848E0(*(_QWORD *)(a1 + 24), *(_QWORD *)(a2 - 48), 0, 0) )
        {
          sub_1391610(a1, v3, v18);
        }
      }
      else
      {
        sub_13848E0(*(_QWORD *)(a1 + 24), *(_QWORD *)(a2 - 48), 0, 0);
      }
    }
    else
    {
      v12 = *(_QWORD *)(a1 + 24);
      v13 = sub_14C81A0(*(_QWORD *)(a2 - 48));
      v14 = v12;
      if ( (unsigned __int8)sub_13848E0(v12, v3, 0, v13) )
      {
        v16 = *(_QWORD *)(a1 + 24);
        v17 = sub_14C8160(v14, v3, v15);
        sub_13848E0(v16, v3, 1u, v17);
      }
    }
    if ( a2 != v3 )
      sub_1391C50(a1, v3, a2, 0);
  }
  result = *(_QWORD *)v4;
  if ( *(_BYTE *)(*(_QWORD *)v4 + 8LL) == 15 )
  {
    result = *(_QWORD *)a2;
    if ( *(_BYTE *)(*(_QWORD *)a2 + 8LL) == 15 )
    {
      v6 = *(_BYTE *)(v4 + 16);
      if ( v6 > 3u )
      {
        if ( v6 == 5 )
        {
          result = (unsigned int)*(unsigned __int16 *)(v4 + 18) - 51;
          if ( (unsigned int)result > 1 )
          {
            result = sub_13848E0(*(_QWORD *)(a1 + 24), v4, 0, 0);
            if ( (_BYTE)result )
              result = sub_1391610(a1, v4, v19);
          }
        }
        else
        {
          result = sub_13848E0(*(_QWORD *)(a1 + 24), v4, 0, 0);
        }
      }
      else
      {
        v7 = *(_QWORD *)(a1 + 24);
        v8 = sub_14C81A0(v4);
        v9 = v7;
        result = sub_13848E0(v7, v4, 0, v8);
        if ( (_BYTE)result )
        {
          v20 = *(_QWORD *)(a1 + 24);
          v21 = sub_14C8160(v9, v4, v10);
          result = sub_13848E0(v20, v4, 1u, v21);
        }
      }
      if ( a2 != v4 )
        return (__int64)sub_1391C50(a1, v4, a2, 0);
    }
  }
  return result;
}
