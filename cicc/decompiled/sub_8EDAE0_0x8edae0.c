// Function: sub_8EDAE0
// Address: 0x8edae0
//
unsigned __int8 *__fastcall sub_8EDAE0(unsigned __int8 *a1, int a2, int a3, __int64 a4)
{
  char v7; // r15
  unsigned __int8 *v8; // r12
  unsigned __int8 v9; // al
  char v10; // al
  unsigned __int8 *v12; // rax
  int v13; // esi
  char v14; // di
  __int64 v15; // rdx
  unsigned __int64 v16; // rax
  int v17; // eax
  __int64 v18; // rcx
  unsigned __int64 v19; // rdx
  unsigned __int64 v20; // rsi
  unsigned __int8 *v21; // rax
  unsigned __int64 v22; // rcx
  int v23; // [rsp+0h] [rbp-70h] BYREF
  int v24; // [rsp+4h] [rbp-6Ch]
  int v25; // [rsp+8h] [rbp-68h]
  _BYTE *v26; // [rsp+10h] [rbp-60h]
  _BYTE v27[80]; // [rsp+20h] [rbp-50h] BYREF

  if ( a3 )
  {
    v7 = a3;
    v12 = sub_8E9510(a1, (__int64)&v23, 1, a4);
    ++*(_QWORD *)(a4 + 32);
    v8 = v12;
  }
  else
  {
    ++*(_QWORD *)(a4 + 48);
    v7 = 2;
    v8 = sub_8E9510(a1, (__int64)&v23, 2, a4);
  }
  v9 = *v8;
  if ( *v8 != 69 && v9 )
  {
    if ( v9 == 81 )
    {
      if ( !*(_QWORD *)(a4 + 32) )
        sub_8E5790(" [overriding ", a4);
      v8 = sub_8E9510(v8 + 1, (__int64)v27, v7, a4);
      if ( !*(_QWORD *)(a4 + 32) )
        sub_8E5790((unsigned __int8 *)"] ", a4);
    }
    if ( a3 )
      --*(_QWORD *)(a4 + 32);
    v13 = v23;
    if ( a2 )
    {
      v8 = sub_8EBA20(v8, v23, v7, a4);
    }
    else
    {
      ++*(_QWORD *)(a4 + 32);
      v21 = sub_8EBA20(v8, v13, v7, a4);
      --*(_QWORD *)(a4 + 32);
      v8 = v21;
    }
    if ( a3 )
      ++*(_QWORD *)(a4 + 32);
    v14 = v24;
    if ( v24 )
    {
      if ( !*(_QWORD *)(a4 + 32) )
      {
        v15 = *(_QWORD *)(a4 + 8);
        v16 = v15 + 1;
        if ( !*(_DWORD *)(a4 + 28) )
        {
          v22 = *(_QWORD *)(a4 + 16);
          if ( v22 > v16 )
          {
            *(_BYTE *)(*(_QWORD *)a4 + v15) = 32;
            v14 = v24;
            v16 = *(_QWORD *)(a4 + 8) + 1LL;
          }
          else
          {
            *(_DWORD *)(a4 + 28) = 1;
            if ( v22 )
            {
              *(_BYTE *)(*(_QWORD *)a4 + v22 - 1) = 0;
              v14 = v24;
              v16 = *(_QWORD *)(a4 + 8) + 1LL;
            }
          }
        }
        *(_QWORD *)(a4 + 8) = v16;
      }
      sub_8E6E80(v14, 0, a4);
    }
    v17 = v25;
    if ( v25 && !*(_QWORD *)(a4 + 32) )
    {
      v18 = *(_QWORD *)(a4 + 8);
      v19 = v18 + 1;
      if ( !*(_DWORD *)(a4 + 28) )
      {
        v20 = *(_QWORD *)(a4 + 16);
        if ( v20 > v19 )
        {
          *(_BYTE *)(*(_QWORD *)a4 + v18) = 32;
          v17 = v25;
          v19 = *(_QWORD *)(a4 + 8) + 1LL;
        }
        else
        {
          *(_DWORD *)(a4 + 28) = 1;
          if ( v20 )
          {
            *(_BYTE *)(*(_QWORD *)a4 + v20 - 1) = 0;
            v17 = v25;
            v19 = *(_QWORD *)(a4 + 8) + 1LL;
          }
        }
      }
      *(_QWORD *)(a4 + 8) = v19;
      if ( v17 == 1 )
      {
        if ( !*(_QWORD *)(a4 + 32) )
          sub_8E5790((unsigned __int8 *)"&", a4);
      }
      else if ( (v17 & 2) != 0 && !*(_QWORD *)(a4 + 32) )
      {
        sub_8E5790((unsigned __int8 *)"&&", a4);
      }
    }
  }
  if ( v26 )
  {
    switch ( *v26 )
    {
      case '0':
        if ( !*(_QWORD *)(a4 + 32) )
          sub_8E5790(" [deleting]", a4);
        break;
      case '1':
      case '7':
        break;
      case '2':
        if ( !*(_QWORD *)(a4 + 32) )
          sub_8E5790(" [subobject]", a4);
        break;
      case '3':
        if ( !*(_QWORD *)(a4 + 32) )
          sub_8E5790(" [allocating]", a4);
        break;
      case '8':
        if ( !*(_QWORD *)(a4 + 32) )
          sub_8E5790(" [static]", a4);
        break;
      case '9':
        if ( !*(_QWORD *)(a4 + 32) )
          sub_8E5790(" [delegation]", a4);
        break;
      case 'I':
        v10 = v26[1];
        if ( v10 == 49 )
        {
          if ( !*(_QWORD *)(a4 + 32) )
            sub_8E5790(" [complete inheriting]", a4);
        }
        else
        {
          if ( v10 != 50 )
            goto LABEL_9;
          if ( !*(_QWORD *)(a4 + 32) )
            sub_8E5790(" [base inheriting]", a4);
        }
        break;
      default:
LABEL_9:
        if ( !*(_DWORD *)(a4 + 24) )
        {
          ++*(_QWORD *)(a4 + 32);
          ++*(_QWORD *)(a4 + 48);
          *(_DWORD *)(a4 + 24) = 1;
        }
        break;
    }
  }
  if ( a3 )
    --*(_QWORD *)(a4 + 32);
  else
    --*(_QWORD *)(a4 + 48);
  return v8;
}
