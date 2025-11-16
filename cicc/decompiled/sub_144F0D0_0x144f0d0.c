// Function: sub_144F0D0
// Address: 0x144f0d0
//
unsigned __int64 __fastcall sub_144F0D0(__int64 a1)
{
  __int64 v2; // r14
  _QWORD *v3; // r15
  __int64 v4; // rdi
  unsigned __int64 result; // rax
  int v6; // esi
  _QWORD *j; // rax
  __int64 v8; // rbx
  unsigned int v9; // r13d
  __int64 v10; // rdi
  int v11; // eax
  unsigned __int64 v12; // rax
  __int64 v13; // rbx
  __int64 *v14; // rax
  char v15; // dl
  __int64 *v16; // rsi
  unsigned int v17; // r8d
  __int64 *v18; // rdi
  unsigned int v19; // r13d
  __int64 v20; // rbx
  __int64 i; // rdi
  int v22; // eax
  __int64 v23; // [rsp+8h] [rbp-78h]
  unsigned int v24; // [rsp+14h] [rbp-6Ch]
  __int64 v25; // [rsp+18h] [rbp-68h]
  __int64 v26; // [rsp+18h] [rbp-68h]
  __int64 v27[4]; // [rsp+20h] [rbp-60h] BYREF
  char v28; // [rsp+40h] [rbp-40h]

  v2 = *(_QWORD *)(a1 + 112);
  v3 = *(_QWORD **)(v2 - 40);
  if ( !*(_BYTE *)(v2 - 8) )
    goto LABEL_27;
  while ( 1 )
  {
    while ( 1 )
    {
      while ( 1 )
      {
        v4 = sub_157EBA0(*v3 & 0xFFFFFFFFFFFFFFF8LL);
        result = 0;
        if ( v4 )
          result = sub_15F4D60(v4);
        v24 = *(_DWORD *)(v2 - 16);
        v25 = *(_QWORD *)(v2 - 32);
        if ( v24 != (_DWORD)result || v3 != *(_QWORD **)(v2 - 32) )
          break;
        *(_QWORD *)(a1 + 112) -= 40LL;
        v2 = *(_QWORD *)(a1 + 112);
        if ( v2 == *(_QWORD *)(a1 + 104) )
          return result;
        v3 = *(_QWORD **)(v2 - 40);
        if ( !*(_BYTE *)(v2 - 8) )
        {
LABEL_27:
          v19 = 0;
          v20 = sub_157EBA0(*v3 & 0xFFFFFFFFFFFFFFF8LL);
          for ( i = v20; ; i = sub_157EBA0(*v3 & 0xFFFFFFFFFFFFFFF8LL) )
          {
            v22 = 0;
            if ( i )
              v22 = sub_15F4D60(i);
            if ( v22 == v19 )
              break;
            v26 = *(_QWORD *)(v3[1] + 32LL);
            if ( v26 != sub_15F4DF0(v20, v19) )
              break;
            ++v19;
          }
          *(_BYTE *)(v2 - 8) = 1;
          *(_QWORD *)(v2 - 32) = v3;
          *(_QWORD *)(v2 - 24) = v20;
          *(_DWORD *)(v2 - 16) = v19;
        }
      }
      v6 = *(_DWORD *)(v2 - 16);
      v23 = *(_QWORD *)(v2 - 24);
      for ( j = *(_QWORD **)(v2 - 32); ; j = *(_QWORD **)(v2 - 32) )
      {
        v9 = v6 + 1;
        *(_DWORD *)(v2 - 16) = v6 + 1;
        v10 = sub_157EBA0(*j & 0xFFFFFFFFFFFFFFF8LL);
        v11 = 0;
        if ( v10 )
        {
          v11 = sub_15F4D60(v10);
          v9 = *(_DWORD *)(v2 - 16);
        }
        if ( v11 == v9 )
          break;
        v8 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v2 - 32) + 8LL) + 32LL);
        if ( v8 != sub_15F4DF0(*(_QWORD *)(v2 - 24), v9) )
          break;
        v6 = *(_DWORD *)(v2 - 16);
      }
      v12 = sub_15F4DF0(v23, v24);
      v13 = sub_1444DB0(*(_QWORD **)(v25 + 8), v12);
      v14 = *(__int64 **)(a1 + 8);
      if ( *(__int64 **)(a1 + 16) != v14 )
        goto LABEL_12;
      v16 = &v14[*(unsigned int *)(a1 + 28)];
      v17 = *(_DWORD *)(a1 + 28);
      if ( v14 == v16 )
        break;
      v18 = 0;
      while ( v13 != *v14 )
      {
        if ( *v14 == -2 )
        {
          v18 = v14;
          if ( v14 + 1 == v16 )
            goto LABEL_21;
          ++v14;
        }
        else if ( v16 == ++v14 )
        {
          if ( !v18 )
            goto LABEL_34;
LABEL_21:
          *v18 = v13;
          --*(_DWORD *)(a1 + 32);
          ++*(_QWORD *)a1;
          goto LABEL_13;
        }
      }
    }
LABEL_34:
    if ( v17 < *(_DWORD *)(a1 + 24) )
      break;
LABEL_12:
    sub_16CCBA0(a1, v13);
    if ( v15 )
      goto LABEL_13;
  }
  *(_DWORD *)(a1 + 28) = v17 + 1;
  *v16 = v13;
  ++*(_QWORD *)a1;
LABEL_13:
  v27[0] = v13;
  v28 = 0;
  return sub_144DD80((__int64 *)(a1 + 104), (__int64)v27);
}
