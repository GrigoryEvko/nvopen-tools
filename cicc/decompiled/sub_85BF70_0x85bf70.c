// Function: sub_85BF70
// Address: 0x85bf70
//
__int64 __fastcall sub_85BF70(__int64 a1)
{
  __int64 **v1; // rbx
  __int64 result; // rax
  __int64 *v3; // r12
  __int64 v5; // rsi
  __int64 v6; // rdi
  __int64 v7; // rcx
  char v8; // dl
  __int64 v9; // rdx

  v1 = *(__int64 ***)(*(_QWORD *)(a1 + 8) + 24LL);
  result = *(_QWORD *)(a1 + 16);
  v3 = *(__int64 **)(result + 8);
  if ( v1 )
  {
    while ( 1 )
    {
      result = *((unsigned int *)v1 + 8);
      if ( !(_DWORD)result )
        break;
      if ( (_DWORD)result == 4 || (result = v3[5]) == 0 )
      {
LABEL_5:
        v1 = (__int64 **)*v1;
        v3 = (__int64 *)*v3;
        if ( !v1 )
          return result;
      }
      else
      {
        *(_QWORD *)(result + 88) = v3[10];
        v1 = (__int64 **)*v1;
        v3 = (__int64 *)*v3;
        if ( !v1 )
          return result;
      }
    }
    v5 = v3[10];
    v6 = (__int64)v1[1];
    if ( v5 )
    {
      result = sub_85BC00(v6, v5);
    }
    else if ( *(_BYTE *)(a1 + 40) && *(_DWORD *)(a1 + 48) && !*((_BYTE *)v3 + 96) )
    {
      v7 = v3[9];
      result = *(_QWORD *)(v7 + 8);
      v8 = *(_BYTE *)(result + 80);
      if ( v8 == 3 || v8 == 2 )
      {
        *(_QWORD *)(result + 88) = *(_QWORD *)(v7 + 64);
      }
      else
      {
        v9 = *(_QWORD *)(result + 88);
        *(_QWORD *)(v9 + 200) = result;
        *(_QWORD *)(v9 + 208) = 0;
      }
      *(_BYTE *)(result + 82) &= ~1u;
    }
    else
    {
      result = sub_85B8C0(v6, 0);
    }
    goto LABEL_5;
  }
  return result;
}
