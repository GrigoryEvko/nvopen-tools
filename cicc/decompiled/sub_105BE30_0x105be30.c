// Function: sub_105BE30
// Address: 0x105be30
//
__int64 __fastcall sub_105BE30(__int64 a1)
{
  __int64 v1; // rax
  __int64 *v2; // r14
  __int64 v3; // r13
  __int64 v4; // rax
  unsigned __int8 *v5; // r12
  __int64 *v6; // rdx
  __int64 v7; // rcx
  __int64 v8; // r8
  __int64 v9; // r9
  __int64 *v11; // r12
  __int64 *v12; // rbx
  __int64 v13; // rdx
  __int64 v14; // r8
  __int64 v15; // r9
  __int64 v16; // rcx
  __int64 v17; // rsi
  unsigned int v18; // [rsp+Ch] [rbp-34h]

  sub_C7D6A0(0, 0, 8);
  v1 = *(unsigned int *)(a1 + 264);
  if ( (_DWORD)v1 )
  {
    v3 = 8 * v1;
    v2 = (__int64 *)sub_C7D670(8 * v1, 8);
    v18 = *(_DWORD *)(a1 + 256);
    v11 = &v2[(unsigned __int64)v3 / 8];
    v12 = v2;
    memcpy(v2, *(const void **)(a1 + 248), v3);
    v16 = v18;
    if ( v18 )
    {
      while ( *v12 == -8192 || *v12 == -4096 )
      {
        if ( v11 == ++v12 )
          goto LABEL_3;
      }
      while ( v12 != v11 )
      {
        v17 = *v12++;
        sub_1058070(a1, v17, v13, v16, v14, v15);
        if ( v12 == v11 )
          break;
        while ( *v12 == -8192 || *v12 == -4096 )
        {
          if ( v11 == ++v12 )
            goto LABEL_3;
        }
      }
    }
  }
  else
  {
    v2 = 0;
    v3 = 0;
  }
LABEL_3:
  while ( 1 )
  {
    v4 = *(_QWORD *)(a1 + 568);
    if ( v4 == *(_QWORD *)(a1 + 560) )
      break;
    while ( 1 )
    {
      v5 = *(unsigned __int8 **)(v4 - 8);
      *(_QWORD *)(a1 + 568) = v4 - 8;
      sub_1056340(a1, v5);
      if ( (unsigned int)*v5 - 30 <= 0xA )
        break;
      sub_1058110(a1, (__int64)v5, (__int64)v6, v7, v8, v9);
      v4 = *(_QWORD *)(a1 + 568);
      if ( v4 == *(_QWORD *)(a1 + 560) )
        return sub_C7D6A0((__int64)v2, v3, 8);
    }
    sub_105B420(a1, (__int64)v5, v6, v7, v8, v9);
  }
  return sub_C7D6A0((__int64)v2, v3, 8);
}
