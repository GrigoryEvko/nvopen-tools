// Function: sub_2F54E60
// Address: 0x2f54e60
//
__int64 __fastcall sub_2F54E60(__int64 a1, int a2, __int64 a3)
{
  __int64 v4; // rdx
  __int64 result; // rax
  __int64 v6; // rbx
  __int64 v7; // r12
  __int64 v8; // rsi
  __int64 (__fastcall *v9)(__int64); // rax
  _DWORD *v10; // rax
  _DWORD *v11; // rdx
  __int64 v12; // rax
  unsigned int v13; // r13d
  unsigned int v14; // ecx
  __int64 v15; // r8
  __int64 v16; // r9
  __int64 v17; // rax
  unsigned __int64 v18; // r12
  __int64 *v19; // rax
  unsigned int v21; // [rsp+8h] [rbp-58h]
  __int64 v22; // [rsp+8h] [rbp-58h]
  _QWORD v23[2]; // [rsp+10h] [rbp-50h] BYREF
  char v24; // [rsp+20h] [rbp-40h]

  v4 = *(_QWORD *)(a1 + 16);
  if ( a2 < 0 )
  {
    result = *(_QWORD *)(v4 + 56) + 16LL * (a2 & 0x7FFFFFFF);
    v6 = *(_QWORD *)(result + 8);
  }
  else
  {
    result = (unsigned int)a2;
    v6 = *(_QWORD *)(*(_QWORD *)(v4 + 304) + 8LL * (unsigned int)a2);
  }
  if ( v6 )
  {
    if ( (*(_BYTE *)(v6 + 4) & 8) == 0 )
    {
LABEL_5:
      v7 = *(_QWORD *)(v6 + 16);
      if ( *(_WORD *)(v7 + 68) == 20 )
        goto LABEL_16;
      while ( 1 )
      {
        v8 = *(_QWORD *)(a1 + 776);
        v9 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v8 + 520LL);
        if ( v9 == sub_2DCA430 )
          goto LABEL_7;
        ((void (__fastcall *)(_QWORD *, __int64, __int64))v9)(v23, v8, v7);
        v10 = (_DWORD *)v23[0];
        v11 = (_DWORD *)v23[1];
        if ( !v24 )
          break;
        while ( 1 )
        {
          if ( (*v10 & 0xFFF00) == 0 && (*v11 & 0xFFF00) == 0 )
          {
            v12 = *(_QWORD *)(v7 + 32);
            v13 = *(_DWORD *)(v12 + 8);
            if ( v13 != a2 || (v13 = *(_DWORD *)(v12 + 48), v13 != a2) )
            {
              v14 = v13;
              if ( v13 - 1 > 0x3FFFFFFE )
                v14 = *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 24) + 32LL) + 4LL * (v13 & 0x7FFFFFFF));
              v21 = v14;
              v15 = sub_2E39EA0(*(__int64 **)(a1 + 792), *(_QWORD *)(v7 + 24));
              v17 = *(unsigned int *)(a3 + 8);
              v18 = ((unsigned __int64)v21 << 32) | v13;
              if ( v17 + 1 > (unsigned __int64)*(unsigned int *)(a3 + 12) )
              {
                v22 = v15;
                sub_C8D5F0(a3, (const void *)(a3 + 16), v17 + 1, 0x10u, v15, v16);
                v17 = *(unsigned int *)(a3 + 8);
                v15 = v22;
              }
              v19 = (__int64 *)(*(_QWORD *)a3 + 16 * v17);
              *v19 = v15;
              v19[1] = v18;
              ++*(_DWORD *)(a3 + 8);
            }
          }
          result = *(_QWORD *)(v6 + 16);
          while ( 1 )
          {
LABEL_9:
            v6 = *(_QWORD *)(v6 + 32);
            if ( !v6 )
              return result;
            if ( (*(_BYTE *)(v6 + 4) & 8) == 0 )
            {
              v7 = *(_QWORD *)(v6 + 16);
              if ( result != v7 )
                break;
            }
          }
          if ( *(_WORD *)(v7 + 68) != 20 )
            break;
LABEL_16:
          v10 = *(_DWORD **)(v7 + 32);
          v11 = v10 + 10;
        }
      }
      v7 = *(_QWORD *)(v6 + 16);
LABEL_7:
      result = v7;
      goto LABEL_9;
    }
    while ( 1 )
    {
      v6 = *(_QWORD *)(v6 + 32);
      if ( !v6 )
        break;
      if ( (*(_BYTE *)(v6 + 4) & 8) == 0 )
        goto LABEL_5;
    }
  }
  return result;
}
