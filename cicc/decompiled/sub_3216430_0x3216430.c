// Function: sub_3216430
// Address: 0x3216430
//
__int64 __fastcall sub_3216430(__int64 a1, __int64 a2, __int16 a3)
{
  __int64 *v6; // rdi
  __int64 v7; // rax
  void (*v8)(); // rax
  __int64 result; // rax
  __int64 *v10; // rsi
  __int64 *v11; // rbx
  char *v12; // [rsp+0h] [rbp-50h] BYREF
  int v13; // [rsp+10h] [rbp-40h]
  __int16 v14; // [rsp+20h] [rbp-30h]

  if ( *(_BYTE *)(a2 + 488) )
  {
    if ( (unsigned int)(*(_DWORD *)(*(_QWORD *)(a2 + 200) + 544LL) - 42) <= 1 )
    {
      v6 = *(__int64 **)(a2 + 224);
      v7 = *v6;
      v13 = *(_DWORD *)(a1 + 8);
      v12 = "Size: ";
      v8 = *(void (**)())(v7 + 120);
      v14 = 2307;
      if ( v8 != nullsub_98 )
        ((void (__fastcall *)(__int64 *, char **, __int64))v8)(v6, &v12, 1);
    }
  }
  switch ( a3 )
  {
    case 3:
      result = sub_31DC9F0(a2, *(_DWORD *)(a1 + 8));
      goto LABEL_7;
    case 4:
      result = sub_31DCA10(a2, *(_DWORD *)(a1 + 8));
      goto LABEL_7;
    case 9:
    case 24:
      result = (*(__int64 (__fastcall **)(__int64, _QWORD, _QWORD, _QWORD))(*(_QWORD *)a2 + 424LL))(
                 a2,
                 *(unsigned int *)(a1 + 8),
                 0,
                 0);
LABEL_7:
      v10 = *(__int64 **)a1;
      if ( *(_BYTE *)(a2 + 488) )
        goto LABEL_8;
      goto LABEL_13;
    case 10:
      result = sub_31DC9D0(a2, *(_DWORD *)(a1 + 8));
      v10 = *(__int64 **)a1;
      if ( !*(_BYTE *)(a2 + 488) )
        goto LABEL_13;
LABEL_8:
      result = (unsigned int)(*(_DWORD *)(*(_QWORD *)(a2 + 200) + 544LL) - 42);
      if ( (unsigned int)result > 1 )
      {
LABEL_13:
        if ( v10 )
        {
          result = *v10;
          do
          {
            result &= 0xFFFFFFFFFFFFFFF8LL;
            v11 = (__int64 *)result;
            if ( !result )
              break;
            sub_3215FD0(result + 8, a2);
            result = *v11;
          }
          while ( (*v11 & 4) == 0 );
        }
      }
      else
      {
        if ( v10 )
          v10 = (__int64 *)(*v10 & 0xFFFFFFFFFFFFFFF8LL);
        return sub_32160B0(a2, (unsigned __int64)v10, 0);
      }
      return result;
    default:
      BUG();
  }
}
