// Function: sub_283D540
// Address: 0x283d540
//
__int64 __fastcall sub_283D540(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 result; // rax
  unsigned __int64 v8; // rbx
  __int64 v9; // r9
  __int64 v10; // r8
  __int64 v11; // rax
  __int64 v12; // rdi
  bool v13; // zf
  __int64 v14; // rdi
  __int64 (__fastcall *v15)(__int64); // rax
  _BYTE *v16; // rax
  unsigned int v17; // [rsp+4h] [rbp-3Ch]
  unsigned int v18; // [rsp+4h] [rbp-3Ch]
  unsigned int v19; // [rsp+4h] [rbp-3Ch]
  unsigned int v20; // [rsp+8h] [rbp-38h]
  unsigned int v21; // [rsp+8h] [rbp-38h]
  unsigned int v22; // [rsp+8h] [rbp-38h]
  unsigned int v23; // [rsp+Ch] [rbp-34h]

  result = *(unsigned int *)(a1 + 64);
  v23 = result;
  if ( (_DWORD)result )
  {
    v8 = 0;
    v9 = 0;
    v10 = 0;
    while ( 1 )
    {
      v11 = *(_QWORD *)(*(_QWORD *)a1 + 8LL * ((unsigned int)v8 >> 6));
      if ( _bittest64(&v11, v8) )
        break;
      v18 = v9;
      v21 = v10 + 1;
      v14 = *(_QWORD *)(*(_QWORD *)(a1 + 72) + 8 * v10);
      v15 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v14 + 24LL);
      if ( v15 == sub_2302CF0 )
      {
        result = sub_283D540(v14 + 8);
        v8 = (unsigned int)(v8 + 1);
        v10 = v21;
        v9 = v18;
        v13 = (_DWORD)v8 == v23;
        if ( (unsigned int)v8 >= v23 )
          goto LABEL_6;
LABEL_10:
        v16 = *(_BYTE **)(a2 + 32);
        if ( (unsigned __int64)v16 >= *(_QWORD *)(a2 + 24) )
        {
          v19 = v9;
          v22 = v10;
          sub_CB5D20(a2, 44);
          v10 = v22;
          v9 = v19;
        }
        else
        {
          *(_QWORD *)(a2 + 32) = v16 + 1;
          *v16 = 44;
        }
      }
      else
      {
        result = ((__int64 (__fastcall *)(__int64, __int64, __int64, __int64))v15)(v14, a2, a3, a4);
        v10 = v21;
        v9 = v18;
LABEL_5:
        v8 = (unsigned int)(v8 + 1);
        v13 = (_DWORD)v8 == v23;
        if ( (unsigned int)v8 < v23 )
          goto LABEL_10;
LABEL_6:
        if ( v13 )
          return result;
      }
    }
    v17 = v10;
    v20 = v9 + 1;
    v12 = *(_QWORD *)(*(_QWORD *)(a1 + 96) + 8 * v9);
    result = (*(__int64 (__fastcall **)(__int64, __int64, __int64, __int64))(*(_QWORD *)v12 + 24LL))(v12, a2, a3, a4);
    v9 = v20;
    v10 = v17;
    goto LABEL_5;
  }
  return result;
}
