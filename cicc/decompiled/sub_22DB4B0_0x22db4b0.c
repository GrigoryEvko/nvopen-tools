// Function: sub_22DB4B0
// Address: 0x22db4b0
//
unsigned __int64 __fastcall sub_22DB4B0(_QWORD *a1, __int64 a2)
{
  unsigned __int64 result; // rax
  __int64 v3; // r13
  int v4; // r12d
  unsigned int v5; // r15d
  __int64 v6; // r14
  __int64 v7; // r12
  unsigned __int8 *v8; // rdx
  __int64 v9; // r13
  __int64 v10; // rdx
  unsigned __int64 v11; // rcx
  unsigned __int64 v12; // [rsp+8h] [rbp-48h]
  __int64 v13; // [rsp+18h] [rbp-38h]

  if ( !(unsigned __int8)sub_22DB400(a1, a2) )
    sub_C64ED0("Broken region found: enumerated BB not in region!", 1u);
  v12 = *a1 & 0xFFFFFFFFFFFFFFF8LL;
  v13 = a1[4];
  result = *(_QWORD *)(a2 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( result != a2 + 48 )
  {
    if ( !result )
      BUG();
    v3 = result - 24;
    result = (unsigned int)*(unsigned __int8 *)(result - 24) - 30;
    if ( (unsigned int)result <= 0xA )
    {
      result = sub_B46E30(v3);
      v4 = result;
      if ( (_DWORD)result )
      {
        v5 = 0;
        do
        {
          v6 = sub_B46EC0(v3, v5);
          result = sub_22DB400(a1, v6);
          if ( (_BYTE)result != 1 && v13 != v6 )
            sub_C64ED0("Broken region found: edges leaving the region must go to the exit node!", 1u);
          ++v5;
        }
        while ( v4 != v5 );
      }
    }
  }
  if ( v12 != a2 )
  {
    result = a2;
    v7 = *(_QWORD *)(a2 + 16);
    if ( v7 )
    {
      while ( 1 )
      {
        v8 = *(unsigned __int8 **)(v7 + 24);
        result = (unsigned int)*v8 - 30;
        if ( (unsigned __int8)(*v8 - 30) <= 0xAu )
          break;
        v7 = *(_QWORD *)(v7 + 8);
        if ( !v7 )
          return result;
      }
LABEL_15:
      v9 = *((_QWORD *)v8 + 5);
      result = sub_22DB400(a1, v9);
      if ( !(_BYTE)result )
      {
        v10 = a1[3];
        if ( v9 )
        {
          v11 = (unsigned int)(*(_DWORD *)(v9 + 44) + 1);
          result = v11;
        }
        else
        {
          v11 = 0;
          result = 0;
        }
        if ( (unsigned int)result < *(_DWORD *)(v10 + 32) )
        {
          result = *(_QWORD *)(v10 + 24);
          if ( *(_QWORD *)(result + 8 * v11) )
            sub_C64ED0("Broken region found: edges entering the region must go to the entry node!", 1u);
        }
      }
      while ( 1 )
      {
        v7 = *(_QWORD *)(v7 + 8);
        if ( !v7 )
          break;
        v8 = *(unsigned __int8 **)(v7 + 24);
        result = (unsigned int)*v8 - 30;
        if ( (unsigned __int8)(*v8 - 30) <= 0xAu )
          goto LABEL_15;
      }
    }
  }
  return result;
}
