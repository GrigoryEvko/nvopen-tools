// Function: sub_1893470
// Address: 0x1893470
//
__int64 __fastcall sub_1893470(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v3; // rcx
  unsigned int v5; // edx
  _QWORD *v6; // r12
  __int64 v7; // rdi
  __int64 v8; // rax
  int v9; // esi
  __int64 v10; // rax
  __int64 v11; // rcx
  bool v12; // zf
  int v13; // edx
  int v14; // edx
  __int64 v15; // rsi
  __int64 v16; // rcx
  int v17; // r9d
  __int64 v18; // [rsp+8h] [rbp-58h] BYREF
  __int64 (__fastcall **v19)(); // [rsp+10h] [rbp-50h]
  _QWORD v20[2]; // [rsp+18h] [rbp-48h] BYREF
  __int64 v21; // [rsp+28h] [rbp-38h]
  __int64 v22; // [rsp+30h] [rbp-30h]

  result = *(unsigned int *)(a1 + 344);
  v18 = a2;
  if ( (_DWORD)result )
  {
    v3 = *(_QWORD *)(a1 + 328);
    v5 = (result - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v6 = (_QWORD *)(v3 + 48LL * v5);
    v7 = v6[3];
    if ( a2 == v7 )
    {
LABEL_4:
      result = v3 + 48 * result;
      if ( v6 != (_QWORD *)result )
      {
        v8 = sub_220F330(v6[5], a1 + 280);
        v9 = 48;
        j_j___libc_free_0(v8, 48);
        --*(_QWORD *)(a1 + 312);
        v21 = -16;
        v19 = off_49F1D90;
        v22 = 0;
        result = v6[3];
        v20[0] = 2;
        v20[1] = 0;
        if ( result == -16 )
        {
          v6[4] = 0;
        }
        else
        {
          if ( result == -8 || !result )
          {
            v6[3] = -16;
            v14 = v21;
            v16 = v22;
            LOBYTE(result) = v21 != -16;
            LOBYTE(v9) = v21 != 0;
            LOBYTE(v14) = v21 != -8;
            result = v14 & v9 & (unsigned int)result;
          }
          else
          {
            v10 = sub_1649B30(v6 + 1);
            v11 = v21;
            v12 = v21 == -8;
            v6[3] = v21;
            LOBYTE(v10) = v11 != 0;
            LOBYTE(v13) = v11 != -16;
            v16 = v22;
            LOBYTE(v9) = !v12;
            result = v9 & v13 & (unsigned int)v10;
          }
          v6[4] = v16;
          v19 = (__int64 (__fastcall **)())&unk_49EE2B0;
          if ( (_BYTE)result )
            result = sub_1649B30(v20);
        }
        --*(_DWORD *)(a1 + 336);
        v15 = *(_QWORD *)(a1 + 256);
        ++*(_DWORD *)(a1 + 340);
        if ( v15 == *(_QWORD *)(a1 + 264) )
        {
          return sub_18931E0((unsigned __int64 **)(a1 + 248), (char *)v15, &v18);
        }
        else
        {
          if ( v15 )
          {
            result = v18;
            *(_QWORD *)(v15 + 8) = 0;
            *(_QWORD *)v15 = 6;
            *(_QWORD *)(v15 + 16) = result;
            if ( result != 0 && result != -8 && result != -16 )
              result = sub_164C220(v15);
            v15 = *(_QWORD *)(a1 + 256);
          }
          *(_QWORD *)(a1 + 256) = v15 + 24;
        }
      }
    }
    else
    {
      v17 = 1;
      while ( v7 != -8 )
      {
        v5 = (result - 1) & (v17 + v5);
        v6 = (_QWORD *)(v3 + 48LL * v5);
        v7 = v6[3];
        if ( a2 == v7 )
          goto LABEL_4;
        ++v17;
      }
    }
  }
  return result;
}
