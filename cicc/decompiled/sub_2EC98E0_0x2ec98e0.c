// Function: sub_2EC98E0
// Address: 0x2ec98e0
//
__int64 __fastcall sub_2EC98E0(
        _WORD *a1,
        __int16 *a2,
        __int64 a3,
        __int64 a4,
        unsigned __int8 a5,
        __int64 a6,
        __int64 a7)
{
  __int64 result; // rax
  __int64 v11; // r8
  __int16 v12; // ax
  __int64 v13; // rdx
  unsigned int v14; // r11d
  int v15; // r9d
  __int64 (__fastcall *v16)(__int64, __int64, unsigned int); // rcx
  int v17; // esi
  __int64 (__fastcall *v18)(__int64, __int64, unsigned int); // rax
  int v19; // eax
  unsigned int v20; // eax
  int v21; // eax
  int v22; // [rsp+Ch] [rbp-44h]
  unsigned int v23; // [rsp+Ch] [rbp-44h]

  result = sub_2EC9250((unsigned int)(__int16)a1[1] >> 31, (unsigned int)a2[1] >> 31, a3, a4, a5);
  if ( !(_BYTE)result && *(_BYTE *)(a4 + 25) == *(_BYTE *)(a3 + 25) )
  {
    v12 = *a2;
    v13 = (unsigned __int16)(*a1 - 1);
    v14 = (unsigned __int16)(*a2 - 1);
    if ( v14 == (_DWORD)v13 )
    {
      return sub_2EC9220((__int16)a1[1], a2[1], a3, a4, a5);
    }
    else
    {
      v15 = 0x7FFFFFFF;
      if ( *a1 )
      {
        v16 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int))(*(_QWORD *)a6 + 368LL);
        if ( v16 != sub_2EC09C0 )
        {
          v23 = (unsigned __int16)(v12 - 1);
          v21 = ((__int64 (__fastcall *)(__int64, __int64, __int64, __int64 (__fastcall *)(__int64, __int64, unsigned int), __int64, __int64))v16)(
                  a6,
                  a7,
                  v13,
                  v16,
                  v11,
                  0x7FFFFFFF);
          v14 = v23;
          LODWORD(v13) = v21;
          v12 = *a2;
        }
        v15 = v13;
      }
      v17 = 0x7FFFFFFF;
      if ( v12 )
      {
        v18 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int))(*(_QWORD *)a6 + 368LL);
        if ( v18 != sub_2EC09C0 )
        {
          v22 = v15;
          v20 = v18(a6, a7, v14);
          v15 = v22;
          v14 = v20;
        }
        v17 = v14;
      }
      if ( (__int16)a1[1] < 0 )
      {
        v19 = v15;
        v15 = v17;
        v17 = v19;
      }
      return sub_2EC9250(v15, v17, a3, a4, a5);
    }
  }
  return result;
}
