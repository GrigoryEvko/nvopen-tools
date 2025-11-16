// Function: sub_2563500
// Address: 0x2563500
//
__int64 __fastcall sub_2563500(__m128i *a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // r12
  __int64 result; // rax
  __int64 v5; // rax
  __int64 v6; // rax
  __int64 v7; // rax
  __int64 v8; // rax

  switch ( (unsigned __int8)sub_2509800(a1) )
  {
    case 0u:
    case 4u:
    case 5u:
      BUG();
    case 1u:
      v6 = sub_A777F0(0x68u, *(__int64 **)(a2 + 128));
      v3 = v6;
      if ( !v6 )
        goto LABEL_4;
      *(__m128i *)(v6 + 72) = _mm_loadu_si128(a1);
      sub_2553350(v6);
      *(_DWORD *)(v3 + 96) = 458752;
      *(_QWORD *)v3 = off_4A1AEA0;
      *(_QWORD *)(v3 + 88) = &unk_4A1AF28;
      result = v3;
      break;
    case 2u:
      v7 = sub_A777F0(0x68u, *(__int64 **)(a2 + 128));
      v3 = v7;
      if ( v7 )
      {
        *(__m128i *)(v7 + 72) = _mm_loadu_si128(a1);
        sub_2553350(v7);
        *(_DWORD *)(v3 + 96) = 458752;
        *(_QWORD *)v3 = off_4A1AF88;
        *(_QWORD *)(v3 + 88) = &unk_4A1B010;
        BUG();
      }
      goto LABEL_4;
    case 3u:
      v8 = sub_A777F0(0x68u, *(__int64 **)(a2 + 128));
      v3 = v8;
      if ( !v8 )
        goto LABEL_4;
      *(__m128i *)(v8 + 72) = _mm_loadu_si128(a1);
      sub_2553350(v8);
      *(_DWORD *)(v3 + 96) = 458752;
      *(_QWORD *)v3 = off_4A1B070;
      *(_QWORD *)(v3 + 88) = &unk_4A1B0F8;
      result = v3;
      break;
    case 6u:
      v2 = sub_A777F0(0x68u, *(__int64 **)(a2 + 128));
      v3 = v2;
      if ( v2 )
      {
        *(__m128i *)(v2 + 72) = _mm_loadu_si128(a1);
        sub_2553350(v2);
        *(_DWORD *)(v3 + 96) = 458752;
        *(_QWORD *)v3 = off_4A1ACD0;
        *(_QWORD *)(v3 + 88) = &unk_4A1AD58;
      }
      goto LABEL_4;
    case 7u:
      v5 = sub_A777F0(0x68u, *(__int64 **)(a2 + 128));
      v3 = v5;
      if ( !v5 )
        goto LABEL_4;
      *(__m128i *)(v5 + 72) = _mm_loadu_si128(a1);
      sub_2553350(v5);
      *(_DWORD *)(v3 + 96) = 458752;
      *(_QWORD *)v3 = off_4A1ADB8;
      *(_QWORD *)(v3 + 88) = &unk_4A1AE40;
      result = v3;
      break;
    default:
      v3 = 0;
LABEL_4:
      result = v3;
      break;
  }
  return result;
}
