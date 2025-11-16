// Function: sub_26807A0
// Address: 0x26807a0
//
__int64 __fastcall sub_26807A0(__int64 a1, __int64 a2, __int64 (__fastcall *a3)(__int64, _QWORD, __int64), __int64 a4)
{
  __int64 result; // rax
  __int64 v6; // r12
  __int64 v7; // rsi
  _QWORD *v8; // rax
  _QWORD *v9; // r15
  int v10; // r15d
  _QWORD *v11; // rbx
  char i; // al
  __int64 v13; // r8
  __int64 v14; // r9
  __int64 v15; // rax
  unsigned __int64 v16; // rdx
  int v17; // eax
  unsigned int v18; // edx
  __int64 *v20; // [rsp+18h] [rbp-88h]
  _QWORD *v21; // [rsp+20h] [rbp-80h]
  __int64 *v22; // [rsp+30h] [rbp-70h]
  __int64 v23; // [rsp+38h] [rbp-68h]
  _BYTE *v24; // [rsp+40h] [rbp-60h] BYREF
  __int64 v25; // [rsp+48h] [rbp-58h]
  _BYTE v26[80]; // [rsp+50h] [rbp-50h] BYREF

  result = *(_QWORD *)a2 + 8LL * *(unsigned int *)(a2 + 8);
  v20 = (__int64 *)result;
  v22 = *(__int64 **)a2;
  if ( result != *(_QWORD *)a2 )
  {
    while ( 1 )
    {
      v6 = *v22;
      v7 = *v22;
      v24 = v26;
      v25 = 0x800000000LL;
      v8 = sub_267FA80(a1, v7);
      v9 = v8;
      v23 = *v8 + 8LL * *((unsigned int *)v8 + 2);
      if ( *v8 != v23 )
        break;
      while ( 1 )
      {
        v17 = v25;
        if ( !(_DWORD)v25 )
          break;
LABEL_11:
        v18 = *(_DWORD *)&v24[4 * v17 - 4];
        LODWORD(v25) = v17 - 1;
        *(_QWORD *)(*v9 + 8LL * v18) = *(_QWORD *)(*v9 + 8LL * (unsigned int)(*((_DWORD *)v9 + 2))-- - 8);
      }
LABEL_13:
      if ( v24 != v26 )
        _libc_free((unsigned __int64)v24);
      result = (__int64)++v22;
      if ( v20 == v22 )
        return result;
    }
    v21 = v8;
    v10 = 0;
    v11 = (_QWORD *)*v8;
    for ( i = a3(a4, *(_QWORD *)*v8, v6); ; i = a3(a4, *v11, v6) )
    {
      if ( i )
      {
        v15 = (unsigned int)v25;
        v16 = (unsigned int)v25 + 1LL;
        if ( v16 > HIDWORD(v25) )
        {
          sub_C8D5F0((__int64)&v24, v26, v16, 4u, v13, v14);
          v15 = (unsigned int)v25;
        }
        ++v11;
        *(_DWORD *)&v24[4 * v15] = v10++;
        LODWORD(v25) = v25 + 1;
        if ( (_QWORD *)v23 == v11 )
        {
LABEL_10:
          v17 = v25;
          v9 = v21;
          if ( (_DWORD)v25 )
            goto LABEL_11;
          goto LABEL_13;
        }
      }
      else
      {
        ++v10;
        if ( (_QWORD *)v23 == ++v11 )
          goto LABEL_10;
      }
    }
  }
  return result;
}
