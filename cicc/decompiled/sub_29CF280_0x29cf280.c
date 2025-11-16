// Function: sub_29CF280
// Address: 0x29cf280
//
__int64 __fastcall sub_29CF280(__int64 a1, __int64 a2)
{
  __int64 v3; // r10
  __int64 v4; // rdx
  __int64 v5; // r8
  __int64 v6; // rdi
  unsigned int v7; // ecx
  __int64 v8; // r11
  __int64 v10; // rax
  __int64 v11; // rcx
  __int64 v12; // rdi

  v3 = ((*(_DWORD *)(a1 + 4) & 0x7FFFFFFu) >> 1) - 1;
  v4 = v3 >> 2;
  if ( !(v3 >> 2) )
  {
    v10 = ((*(_DWORD *)(a1 + 4) & 0x7FFFFFFu) >> 1) - 1;
LABEL_14:
    switch ( v10 )
    {
      case 2LL:
        v11 = *(_QWORD *)(a1 - 8);
        v12 = v4;
        break;
      case 3LL:
        v12 = v4 + 1;
        v11 = *(_QWORD *)(a1 - 8);
        if ( a2 == *(_QWORD *)(v11 + 32LL * (unsigned int)(2 * (v4 + 1))) )
          goto LABEL_8;
        break;
      case 1LL:
        v11 = *(_QWORD *)(a1 - 8);
        goto LABEL_18;
      default:
        return a1;
    }
    v4 = v12 + 1;
    if ( a2 == *(_QWORD *)(v11 + 32LL * (unsigned int)(2 * (v12 + 1))) )
    {
      v4 = v12;
      goto LABEL_8;
    }
LABEL_18:
    if ( a2 != *(_QWORD *)(v11 + 32LL * (unsigned int)(2 * v4 + 2)) )
      return a1;
    goto LABEL_8;
  }
  v5 = 4 * v4;
  v6 = *(_QWORD *)(a1 - 8);
  v7 = 2;
  v4 = 0;
  while ( 1 )
  {
    v8 = v4 + 1;
    if ( a2 == *(_QWORD *)(v6 + 32LL * v7) )
      break;
    if ( a2 == *(_QWORD *)(v6 + 32LL * (v7 + 2)) )
      goto LABEL_10;
    v8 = v4 + 3;
    if ( a2 == *(_QWORD *)(v6 + 32LL * (v7 + 4)) )
    {
      v4 += 2;
      break;
    }
    v4 += 4;
    if ( a2 == *(_QWORD *)(v6 + 32LL * (unsigned int)(2 * v4)) )
    {
LABEL_10:
      if ( v3 != v8 )
        return a1;
      return a1;
    }
    v7 += 8;
    if ( v4 == v5 )
    {
      v10 = v3 - v4;
      goto LABEL_14;
    }
  }
LABEL_8:
  if ( v3 != v4 )
    return a1;
  return a1;
}
