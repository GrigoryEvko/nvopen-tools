// Function: sub_20C47C0
// Address: 0x20c47c0
//
__int64 __fastcall sub_20C47C0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 result; // rax
  __int64 v5; // rbx
  __int64 v6; // rdx
  __int64 v7; // rbx
  __int64 v8; // rdx
  __int64 v9; // rax
  __int16 v10; // ax
  __int16 *v11; // r15
  __int64 v12; // rax
  __int64 v13; // rdx
  _BOOL4 v14; // r8d
  __int64 v15; // [rsp+10h] [rbp-70h]
  __int64 v16; // [rsp+20h] [rbp-60h]
  __int64 v17; // [rsp+28h] [rbp-58h]
  _BOOL4 v18; // [rsp+34h] [rbp-4Ch]
  unsigned int v19[13]; // [rsp+4Ch] [rbp-34h] BYREF

  result = *(unsigned int *)(a2 + 40);
  if ( (_DWORD)result )
  {
    v16 = 0;
    v15 = 40 * result;
    do
    {
      v5 = *(_QWORD *)(a2 + 32) + v16;
      if ( !*(_BYTE *)v5
        && ((*(_BYTE *)(v5 + 3) & 0x10) != 0 && (*(_WORD *)(v5 + 2) & 0xFF0) != 0
         || sub_20C2DD0(a1, a2, *(_QWORD *)(a2 + 32) + v16)) )
      {
        v6 = *(_QWORD *)(a1 + 32);
        if ( !v6 )
          BUG();
        v7 = *(unsigned int *)(v5 + 8);
        v8 = *(_QWORD *)(v6 + 56) + 2LL * *(unsigned int *)(*(_QWORD *)(v6 + 8) + 24 * v7 + 4);
LABEL_10:
        v11 = (__int16 *)v8;
        while ( v11 )
        {
          v19[0] = (unsigned __int16)v7;
          v12 = sub_B996D0(a3, v19);
          if ( v13 )
          {
            v14 = 1;
            if ( !v12 && a3 + 8 != v13 )
              v14 = (unsigned int)(unsigned __int16)v7 < *(_DWORD *)(v13 + 32);
            v17 = v13;
            v18 = v14;
            v9 = sub_22077B0(40);
            *(_DWORD *)(v9 + 32) = v19[0];
            sub_220F040(v18, v9, v17, a3 + 8);
            ++*(_QWORD *)(a3 + 40);
          }
          v10 = *v11;
          v8 = 0;
          ++v11;
          LOWORD(v7) = v10 + v7;
          if ( !v10 )
            goto LABEL_10;
        }
      }
      v16 += 40;
      result = v16;
    }
    while ( v16 != v15 );
  }
  return result;
}
