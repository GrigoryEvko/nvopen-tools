// Function: sub_34E7EB0
// Address: 0x34e7eb0
//
__int64 __fastcall sub_34E7EB0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, _QWORD *a5, __int64 a6)
{
  __int64 v6; // r14
  __int64 v7; // r13
  __int64 v8; // r15
  __int64 v10; // r12
  char v11; // r13
  const void *v12; // r15
  size_t v13; // r12
  __int64 v14; // rbx
  __int64 result; // rax
  __int64 v16; // rdi
  __int64 (*v17)(); // rax
  __int64 v18; // rax
  __int64 v19; // r10
  __int64 v20; // rsi
  __int64 v21; // rdi
  __int64 v22; // rsi
  _QWORD *v23; // [rsp+0h] [rbp-60h]
  _QWORD *v25; // [rsp+10h] [rbp-50h]
  __int64 v26; // [rsp+18h] [rbp-48h]
  char v27[49]; // [rsp+2Fh] [rbp-31h] BYREF

  v6 = a4;
  v7 = a2;
  v23 = a5;
  LOBYTE(a4) = a5 != 0;
  v8 = *(_QWORD *)(*(_QWORD *)(a2 + 16) + 56LL);
  v26 = a1 + 560;
  v25 = a5 + 5;
  if ( a3 != v8 )
  {
    v10 = a3;
    v11 = a5 != 0;
    while ( 1 )
    {
      if ( (unsigned __int16)(*(_WORD *)(v8 + 68) - 14) > 4u )
      {
        v16 = *(_QWORD *)(a1 + 528);
        v17 = *(__int64 (**)())(*(_QWORD *)v16 + 920LL);
        if ( v17 == sub_2DB1B30 || !((unsigned __int8 (__fastcall *)(__int64, __int64))v17)(v16, v8) )
        {
          if ( !v11 )
            goto LABEL_18;
          v27[0] = 1;
          v11 = sub_2E8B400(v8, (__int64)v27, a3, a4, a5);
          if ( !v11 )
            goto LABEL_18;
          v18 = *(_QWORD *)(v8 + 32);
          a3 = 5LL * (*(_DWORD *)(v8 + 40) & 0xFFFFFF);
          v19 = v18 + 40LL * (*(_DWORD *)(v8 + 40) & 0xFFFFFF);
          if ( v18 != v19 )
            break;
        }
      }
LABEL_5:
      if ( (*(_BYTE *)v8 & 4) != 0 )
      {
        v8 = *(_QWORD *)(v8 + 8);
        if ( v10 == v8 )
          goto LABEL_7;
      }
      else
      {
        while ( (*(_BYTE *)(v8 + 44) & 8) != 0 )
          v8 = *(_QWORD *)(v8 + 8);
        v8 = *(_QWORD *)(v8 + 8);
        if ( v10 == v8 )
        {
LABEL_7:
          v7 = a2;
          goto LABEL_8;
        }
      }
    }
    a5 = v23;
    while ( 1 )
    {
      if ( *(_BYTE *)v18 )
        goto LABEL_32;
      a4 = *(unsigned int *)(v18 + 8);
      if ( !(_DWORD)a4 || (*(_BYTE *)(v18 + 3) & 0x10) == 0 )
        goto LABEL_32;
      if ( v23[9] )
      {
        a3 = v23[6];
        if ( !a3 )
          goto LABEL_18;
        a6 = (__int64)v25;
        do
        {
          while ( 1 )
          {
            v21 = *(_QWORD *)(a3 + 16);
            v22 = *(_QWORD *)(a3 + 24);
            if ( (unsigned int)a4 <= *(_DWORD *)(a3 + 32) )
              break;
            a3 = *(_QWORD *)(a3 + 24);
            if ( !v22 )
              goto LABEL_41;
          }
          a6 = a3;
          a3 = *(_QWORD *)(a3 + 16);
        }
        while ( v21 );
LABEL_41:
        if ( (_QWORD *)a6 == v25 || (unsigned int)a4 < *(_DWORD *)(a6 + 32) )
          goto LABEL_18;
        v18 += 40;
        if ( v19 == v18 )
          goto LABEL_5;
      }
      else
      {
        a3 = *v23;
        v20 = *v23 + 4LL * *((unsigned int *)v23 + 2);
        if ( *v23 == v20 )
          goto LABEL_18;
        if ( (_DWORD)a4 != *(_DWORD *)a3 )
        {
          while ( 1 )
          {
            a3 += 4;
            if ( v20 == a3 )
              break;
            if ( (_DWORD)a4 == *(_DWORD *)a3 )
              goto LABEL_31;
          }
LABEL_18:
          if ( !(*(unsigned __int8 (__fastcall **)(_QWORD, __int64, _QWORD, _QWORD))(**(_QWORD **)(a1 + 528) + 968LL))(
                  *(_QWORD *)(a1 + 528),
                  v8,
                  *(_QWORD *)v6,
                  *(unsigned int *)(v6 + 8)) )
            BUG();
          v11 = 0;
          sub_34E7040(v8, v26);
          goto LABEL_5;
        }
LABEL_31:
        if ( v20 == a3 )
          goto LABEL_18;
LABEL_32:
        v18 += 40;
        if ( v19 == v18 )
          goto LABEL_5;
      }
    }
  }
LABEL_8:
  v12 = *(const void **)v6;
  v13 = 40LL * *(unsigned int *)(v6 + 8);
  v14 = *(unsigned int *)(v6 + 8);
  result = *(unsigned int *)(v7 + 224);
  if ( v14 + result > (unsigned __int64)*(unsigned int *)(v7 + 228) )
  {
    sub_C8D5F0(v7 + 216, (const void *)(v7 + 232), v14 + result, 0x28u, (__int64)a5, a6);
    result = *(unsigned int *)(v7 + 224);
  }
  if ( v13 )
  {
    memcpy((void *)(*(_QWORD *)(v7 + 216) + 40 * result), v12, v13);
    result = *(unsigned int *)(v7 + 224);
  }
  *(_BYTE *)v7 &= ~4u;
  *(_DWORD *)(v7 + 224) = result + v14;
  *(_DWORD *)(v7 + 4) = 0;
  return result;
}
