// Function: sub_1D5EF60
// Address: 0x1d5ef60
//
__int64 __fastcall sub_1D5EF60(_QWORD *a1, _QWORD *a2)
{
  int v3; // eax
  int v4; // edx
  _QWORD *v5; // rax
  __int64 (*v6)(); // r15
  _QWORD **v7; // rax
  unsigned int v8; // eax
  __int64 v9; // rdx
  __int64 v10; // rbx
  unsigned int v11; // r14d
  unsigned int v12; // eax
  __int64 v13; // rdx
  __int64 result; // rax
  __int64 (*v15)(); // rcx
  _QWORD **v16; // rsi
  __int64 (*v17)(); // rdx

  v3 = *((unsigned __int8 *)a2 + 16);
  if ( v3 != 62 )
  {
    v4 = v3 - 24;
    v5 = (_QWORD *)*a1;
    if ( v4 == 44 )
    {
      v6 = (__int64 (*)())v5[110];
      if ( (*((_BYTE *)a2 + 23) & 0x40) != 0 )
        v7 = (_QWORD **)*(a2 - 1);
      else
        v7 = (_QWORD **)&a2[-3 * (*((_DWORD *)a2 + 5) & 0xFFFFFFF)];
      v8 = sub_1F59570(**v7, 0);
      v10 = v9;
      v11 = v8;
      v12 = sub_1F59570(*a2, 0);
      if ( v6 != sub_1D5A410
        && ((unsigned __int8 (__fastcall *)(_QWORD *, _QWORD, __int64, _QWORD, __int64))v6)(a1, v12, v13, v11, v10) )
      {
        return 1;
      }
    }
    else
    {
      v15 = (__int64 (*)())v5[102];
      if ( (*((_BYTE *)a2 + 23) & 0x40) != 0 )
        v16 = (_QWORD **)*(a2 - 1);
      else
        v16 = (_QWORD **)&a2[-3 * (*((_DWORD *)a2 + 5) & 0xFFFFFFF)];
      if ( v15 == sub_1D5A400 )
        goto LABEL_13;
      if ( ((unsigned __int8 (__fastcall *)(_QWORD *, _QWORD, _QWORD))v15)(a1, **v16, *a2) )
        return 1;
    }
  }
  v5 = (_QWORD *)*a1;
LABEL_13:
  v17 = (__int64 (*)())v5[122];
  result = 0;
  if ( v17 != sub_1D5A420 )
    return ((__int64 (__fastcall *)(_QWORD *, _QWORD *))v17)(a1, a2);
  return result;
}
