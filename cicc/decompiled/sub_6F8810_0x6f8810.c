// Function: sub_6F8810
// Address: 0x6f8810
//
__int64 __fastcall sub_6F8810(__int64 *a1, int *a2, _QWORD *a3, __int64 *a4, _QWORD *a5, _DWORD *a6, _QWORD *a7)
{
  __int64 v10; // r13
  __int64 v11; // r15
  __int64 v12; // rcx
  __int64 v13; // r8
  __int64 v14; // r9
  char v15; // al
  int v16; // eax
  __int64 *v17; // rdi
  _BOOL4 v18; // eax
  __int64 result; // rax
  __int64 v20; // rdx
  __int64 v21; // [rsp+0h] [rbp-1F0h]
  _BOOL4 v24; // [rsp+2Ch] [rbp-1C4h] BYREF
  __int64 *v25; // [rsp+30h] [rbp-1C0h] BYREF
  __int64 *v26; // [rsp+38h] [rbp-1B8h] BYREF
  _BYTE v27[432]; // [rsp+40h] [rbp-1B0h] BYREF

  v10 = *a1;
  v11 = sub_6E3DA0(*a1, (__int64)v27);
  if ( (unsigned int)sub_6E3E30(v10, &v24, (__int64 *)&v26, &v25) )
  {
    v18 = v24;
    goto LABEL_9;
  }
  v15 = *(_BYTE *)(v10 + 24);
  if ( (unsigned __int8)(v15 - 14) > 1u && v15 != 12 )
  {
    if ( v15 == 11 )
    {
      v20 = *(_QWORD *)(v10 + 56);
      v26 = *(__int64 **)(v20 + 16);
      v24 = v26 == 0;
      if ( !v26 )
      {
        v17 = *(__int64 **)(v20 + 56);
        v16 = 1;
        v25 = v17;
LABEL_6:
        *a2 = v16;
        if ( !v17 )
        {
LABEL_7:
          *a4 = 0;
          goto LABEL_12;
        }
        goto LABEL_11;
      }
      goto LABEL_23;
    }
    if ( v15 != 1 )
    {
      if ( v15 == 2 )
      {
        v21 = *(_QWORD *)(v10 + 56);
        v26 = (__int64 *)sub_72F1F0(v21);
        v24 = v26 == 0;
        if ( !v26 )
        {
          v16 = 1;
          v17 = *(__int64 **)(v21 + 184);
          v25 = v17;
          goto LABEL_6;
        }
LABEL_23:
        *a2 = 0;
LABEL_24:
        v17 = v26;
        goto LABEL_16;
      }
LABEL_27:
      sub_721090(v10);
    }
    if ( *(_BYTE *)(v10 + 56) != 24 )
      goto LABEL_27;
    v26 = *(__int64 **)(v10 + 72);
    v18 = v24;
LABEL_9:
    v12 = (__int64)a2;
    *a2 = v18;
    if ( v18 )
    {
      LODWORD(v17) = (_DWORD)v25;
      if ( !v25 )
        goto LABEL_7;
LABEL_11:
      *a4 = sub_6E3F00((int)v17, (__int64)a1, v11);
      goto LABEL_12;
    }
    goto LABEL_24;
  }
  v16 = *(unsigned __int8 *)(v10 + 56);
  v17 = *(__int64 **)(v10 + 64);
  v24 = v16;
  if ( v16 )
  {
    v25 = v17;
    goto LABEL_6;
  }
  v26 = v17;
  *a2 = 0;
LABEL_16:
  sub_6F8800(v17, (__int64)a1, a3, v12, v13, v14);
  *a4 = 0;
LABEL_12:
  *a5 = *(_QWORD *)(v11 + 356);
  result = *(unsigned int *)(v11 + 364);
  *a6 = result;
  if ( a7 )
  {
    result = *(_QWORD *)(v11 + 368);
    *a7 = result;
  }
  return result;
}
