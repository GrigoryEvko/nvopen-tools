// Function: sub_BE0500
// Address: 0xbe0500
//
__int64 *__fastcall sub_BE0500(__int64 **a1, _BYTE *a2, const char *a3, __int64 a4, char a5)
{
  unsigned int v7; // ebx
  __int64 v8; // r14
  unsigned int v9; // ebx
  unsigned __int8 v10; // al
  __int64 v11; // rax
  __int64 *result; // rax
  __int64 v13; // rax
  __int64 v14; // rbx
  __int64 v15; // rdx
  unsigned __int8 v16; // al
  __int64 v17; // rax
  __int64 *v18; // rdi
  __int64 v19; // r12
  _BYTE *v20; // rax
  unsigned int v21; // [rsp+4h] [rbp-8Ch]
  __int64 v23; // [rsp+18h] [rbp-78h]
  unsigned __int8 v24; // [rsp+23h] [rbp-6Dh]
  int v25; // [rsp+24h] [rbp-6Ch]
  unsigned int v26; // [rsp+28h] [rbp-68h]
  __int64 *v27; // [rsp+28h] [rbp-68h]
  const char *v28; // [rsp+30h] [rbp-60h] BYREF
  char v29; // [rsp+50h] [rbp-40h]
  char v30; // [rsp+51h] [rbp-3Fh]

  v24 = *(a3 - 16);
  if ( (v24 & 2) != 0 )
  {
    v26 = *((_DWORD *)a3 - 6);
    if ( v26 != 2 )
      goto LABEL_3;
    v13 = *((_QWORD *)a3 - 4);
    return *(__int64 **)(v13 + 8);
  }
  v26 = (*((_WORD *)a3 - 8) >> 6) & 0xF;
  if ( v26 == 2 )
  {
    v13 = (__int64)&a3[-8 * ((v24 >> 2) & 0xF) - 16];
    return *(__int64 **)(v13 + 8);
  }
LABEL_3:
  v21 = a5 == 0 ? 1 : 3;
  v7 = v21;
  v25 = 3 - (a5 == 0);
  if ( v21 >= v26 )
  {
LABEL_17:
    v14 = v26 - v25;
    if ( (v24 & 2) != 0 )
      v15 = *((_QWORD *)a3 - 4);
    else
      v15 = (__int64)&a3[-8 * ((v24 >> 2) & 0xF) - 16];
    sub_C46B40(a4, *(_QWORD *)(*(_QWORD *)(v15 + 8LL * (unsigned int)(v14 + 1)) + 136LL) + 24LL);
    v16 = *(a3 - 16);
    if ( (v16 & 2) != 0 )
      v17 = *((_QWORD *)a3 - 4);
    else
      v17 = (__int64)&a3[-8 * ((v16 >> 2) & 0xF) - 16];
    return *(__int64 **)(v17 + 8 * v14);
  }
  else
  {
    v23 = (__int64)&a3[-8 * ((v24 >> 2) & 0xF) - 16];
    while ( 1 )
    {
      v8 = v23;
      if ( (v24 & 2) != 0 )
        v8 = *((_QWORD *)a3 - 4);
      if ( (int)sub_C49970(*(_QWORD *)(*(_QWORD *)(v8 + 8LL * (v7 + 1)) + 136LL) + 24LL, a4) > 0 )
        break;
      v7 += v25;
      if ( v7 >= v26 )
        goto LABEL_17;
    }
    if ( v7 == v21 )
    {
      result = *a1;
      if ( *a1 )
      {
        v18 = *a1;
        v27 = *a1;
        v30 = 1;
        v28 = "Could not find TBAA parent in struct type node";
        v29 = 3;
        sub_BDBF70(v18, (__int64)&v28);
        if ( *v27 )
        {
          sub_BDBD80((__int64)v27, a2);
          sub_BD9900(v27, a3);
          v19 = *v27;
          sub_C49420(a4, *v27, 1);
          v20 = *(_BYTE **)(v19 + 32);
          if ( (unsigned __int64)v20 >= *(_QWORD *)(v19 + 24) )
          {
            sub_CB5D20(v19, 10);
            return 0;
          }
          *(_QWORD *)(v19 + 32) = v20 + 1;
          *v20 = 10;
        }
        return 0;
      }
    }
    else
    {
      v9 = v7 - v25;
      sub_C46B40(a4, *(_QWORD *)(*(_QWORD *)(v8 + 8LL * (v9 + 1)) + 136LL) + 24LL);
      v10 = *(a3 - 16);
      if ( (v10 & 2) != 0 )
        v11 = *((_QWORD *)a3 - 4);
      else
        v11 = (__int64)&a3[-8 * ((v10 >> 2) & 0xF) - 16];
      return *(__int64 **)(v11 + 8LL * v9);
    }
  }
  return result;
}
