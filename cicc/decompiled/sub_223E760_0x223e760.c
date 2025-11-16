// Function: sub_223E760
// Address: 0x223e760
//
__int64 *__fastcall sub_223E760(__int64 *a1, __int64 a2)
{
  __int64 v3; // rdx
  __int64 v4; // rcx
  char *v5; // r13
  __int64 v6; // rbp
  __int64 v7; // r8
  char v8; // dl
  char *v9; // rbx
  __int64 v10; // rdi
  _BYTE *v12; // r15
  char v13; // al
  __int64 v14; // rax
  __int64 v15; // rcx
  __int64 (__fastcall *v16)(__int64, unsigned int); // rdx
  char v17[8]; // [rsp+0h] [rbp-48h] BYREF
  _QWORD *v18; // [rsp+8h] [rbp-40h]

  sub_223DFF0((__int64)v17, a1);
  if ( v17[0] )
  {
    v5 = (char *)a1 + *(_QWORD *)(*a1 - 24);
    v6 = *((_QWORD *)v5 + 31);
    if ( !v6 )
      sub_426219(v17, a1, v3, v4);
    if ( v5[225] )
    {
      v7 = (unsigned int)v5[224];
    }
    else
    {
      v12 = (_BYTE *)*((_QWORD *)v5 + 30);
      if ( !v12 )
        sub_426219(v17, a1, v3, v4);
      if ( v12[56] )
      {
        v7 = (unsigned int)(char)v12[89];
        v13 = v12[89];
      }
      else
      {
        sub_2216D60(*((_QWORD *)v5 + 30));
        v7 = 32;
        v16 = *(__int64 (__fastcall **)(__int64, unsigned int))(*(_QWORD *)v12 + 48LL);
        v13 = 32;
        if ( v16 != sub_CE72A0 )
        {
          v13 = ((__int64 (__fastcall *)(_BYTE *, __int64, __int64 (__fastcall *)(__int64, unsigned int), __int64, __int64))v16)(
                  v12,
                  32,
                  v16,
                  v15,
                  32);
          v7 = (unsigned int)v13;
        }
      }
      v5[224] = v13;
      v14 = *a1;
      v5[225] = 1;
      v5 = (char *)a1 + *(_QWORD *)(v14 - 24);
    }
    (*(void (__fastcall **)(__int64, _QWORD, bool, char *, __int64, __int64))(*(_QWORD *)v6 + 32LL))(
      v6,
      *((_QWORD *)v5 + 29),
      *((_QWORD *)v5 + 29) == 0,
      v5,
      v7,
      a2);
    if ( v8 )
      sub_222DC80((__int64)a1 + *(_QWORD *)(*a1 - 24), *(_DWORD *)((char *)a1 + *(_QWORD *)(*a1 - 24) + 32) | 1);
  }
  v9 = (char *)v18 + *(_QWORD *)(*v18 - 24LL);
  if ( (v9[25] & 0x20) != 0 && !(unsigned __int8)sub_2252910() )
  {
    v10 = *((_QWORD *)v9 + 29);
    if ( v10 )
    {
      if ( (*(unsigned int (__fastcall **)(__int64))(*(_QWORD *)v10 + 48LL))(v10) == -1 )
        sub_222DC80(
          (__int64)v18 + *(_QWORD *)(*v18 - 24LL),
          *(_DWORD *)((char *)v18 + *(_QWORD *)(*v18 - 24LL) + 32) | 1);
    }
  }
  return a1;
}
